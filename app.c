/**
  ******************************************************************************
  * @file    app.c
  * @author  MDG Application Team
  * @brief   Sanchaar Mitra - ASL Hand Tracking & UART Streaming (3D Enabled)
  ******************************************************************************
  ******************************************************************************
  Ankur Majumdar - 0110 0x41 0x4D
  */

#include "app.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "app_cam.h"
#include "app_config.h"
#include "IPL_resize.h"
#include "app_postprocess.h"
#include "isp_api.h"
#include "ld.h"
#include "ll_aton_runtime.h"
#include "cmw_camera.h"
#include "scrl.h"
#ifdef STM32N6570_DK_REV
#include "stm32n6570_discovery.h"
#else
#include "stm32n6xx_nucleo.h"
#endif
#include "stm32_lcd.h"
#include "stm32_lcd_ex.h"
#include "stm32n6xx_hal.h"
#include "FreeRTOS.h"
#include "task.h"
#include "semphr.h"
#include "utils.h"

/* --- 1. MACROS & DEFINITIONS --- */
#define FREERTOS_PRIORITY(p) ((UBaseType_t)((int)tskIDLE_PRIORITY + configMAX_PRIORITIES / 2 + (p)))

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#if HAS_ROTATION_SUPPORT == 1
#include "nema_core.h"
#include "nema_error.h"
void nema_enable_tiling(int);
#endif

#define LCD_FG_WIDTH LCD_BG_WIDTH
#define LCD_FG_HEIGHT LCD_BG_HEIGHT

#define CACHE_OP(__op__) do { \
  if (is_cache_enable()) { \
    __op__; \
  } \
} while (0)

#define DBG_INFO 0
#define USE_FILTERED_TS 1
#define BQUEUE_MAX_BUFFERS 2
#define CPU_LOAD_HISTORY_DEPTH 8
#define DISPLAY_BUFFER_NB (DISPLAY_DELAY + 2)
#define PD_MAX_HAND_NB 1
#define UTIL_LCD_COLOR_TRANSPARENT 0

#ifdef STM32N6570_DK_REV
#define LCD_FONT Font20
#define DISK_RADIUS 2
#else
#define LCD_FONT Font12
#define DISK_RADIUS 1
#endif

/* --- 2. STRUCTURE DEFINITIONS --- */
#if HAS_ROTATION_SUPPORT == 1
typedef float app_v3_t[3];
#endif

typedef struct {
  float cx; float cy; float w; float h; float rotation;
} roi_t;

typedef struct { uint32_t X0; uint32_t Y0; uint32_t XSize; uint32_t YSize; } Rectangle_TypeDef;

typedef struct {
  SemaphoreHandle_t free; StaticSemaphore_t free_buffer;
  SemaphoreHandle_t ready; StaticSemaphore_t ready_buffer;
  int buffer_nb; uint8_t *buffers[BQUEUE_MAX_BUFFERS];
  int free_idx; int ready_idx;
} bqueue_t;

typedef struct {
  uint64_t current_total; uint64_t current_thread_total;
  uint64_t prev_total; uint64_t prev_thread_total;
  struct { uint64_t total; uint64_t thread; uint32_t tick; } history[CPU_LOAD_HISTORY_DEPTH];
} cpuload_info_t;

typedef struct { int is_valid; pd_pp_box_t pd_hands; roi_t roi; ld_point_t ld_landmarks[LD_LANDMARK_NB]; } hand_info_t;

typedef struct {
  float nn_period_ms; uint32_t pd_ms; uint32_t hl_ms; uint32_t pp_ms; uint32_t disp_ms;
  int is_ld_displayed; int is_pd_displayed; int pd_hand_nb; float pd_max_prob;
  hand_info_t hands[PD_MAX_HAND_NB];
} display_info_t;

typedef struct {
  SemaphoreHandle_t update; StaticSemaphore_t update_buffer;
  SemaphoreHandle_t lock; StaticSemaphore_t lock_buffer;
  display_info_t info;
} display_t;

typedef struct {
  uint32_t nn_in_len; float *prob_out; uint32_t prob_out_len;
  float *boxes_out; uint32_t boxes_out_len;
  pd_model_pp_static_param_t static_param; pd_pp_out_t pd_out;
} pd_model_info_t;

typedef struct {
  uint8_t *nn_in; uint32_t nn_in_len; float *prob_out;
  uint32_t prob_out_len; float *landmarks_out; uint32_t landmarks_out_len;
} hl_model_info_t;

typedef struct {
  Button_TypeDef button_id; int prev_state;
  void (*on_click_handler)(void *cb_args); void *cb_args;
} button_t;

/* --- 3. FORWARD DECLARATIONS --- */
extern UART_HandleTypeDef huart1;
static void decode_ld_landmark(roi_t *roi, ld_point_t *lm, ld_point_t *decoded);
static void nn_thread_fct(void *arg);
static void dp_thread_fct(void *arg);
static void isp_thread_fct(void *arg);
static int is_cache_enable(void);

/* --- 4. GLOBAL VARIABLES --- */
static Rectangle_TypeDef lcd_bg_area = {.XSize = LCD_BG_WIDTH, .YSize = LCD_BG_HEIGHT};
static Rectangle_TypeDef lcd_fg_area = {.XSize = LCD_FG_WIDTH, .YSize = LCD_FG_HEIGHT};
static uint8_t lcd_bg_buffer[DISPLAY_BUFFER_NB][LCD_BG_WIDTH * LCD_BG_HEIGHT * DISPLAY_BPP] ALIGN_32 IN_PSRAM;
static int lcd_bg_buffer_disp_idx = 1; static int lcd_bg_buffer_capt_idx = 0;
static uint8_t lcd_fg_buffer[2][LCD_FG_WIDTH * LCD_FG_HEIGHT* 2] ALIGN_32 IN_PSRAM;
static int lcd_fg_buffer_rd_idx;
static display_t disp = {.info.is_ld_displayed = 1, .info.is_pd_displayed = 0};
static cpuload_info_t cpu_load;
static uint8_t screen_buffer[LCD_BG_WIDTH * LCD_BG_HEIGHT * 2] ALIGN_32 IN_PSRAM;

LL_ATON_DECLARE_NAMED_NN_INSTANCE_AND_INTERFACE(palm_detector);
static roi_t rois[PD_MAX_HAND_NB];
LL_ATON_DECLARE_NAMED_NN_INSTANCE_AND_INTERFACE(hand_landmark);
static ld_point_t ld_landmarks[PD_MAX_HAND_NB][LD_LANDMARK_NB];
static uint32_t frame_event_nb;
static volatile uint32_t frame_event_nb_for_resize;
static uint8_t nn_input_buffers[2][NN_WIDTH * NN_HEIGHT * NN_BPP] ALIGN_32 IN_PSRAM;
static bqueue_t nn_input_queue;

/* Received character from laptop (sign mode) */
static volatile char rx_char = '\0';  /* latest character received from UART */

/* STT (Speech-to-Text) text from laptop */
#define STT_TEXT_MAX 128
static volatile char  stt_text[STT_TEXT_MAX];  /* accumulated STT sentence */
static volatile int   stt_text_len = 0;        /* current length */
static volatile int   stt_active = 0;          /* 1 = showing STT text, 0 = sign mode */

/* Stack Definitions */
static StaticTask_t nn_thread; static StackType_t nn_thread_stack[2 * configMINIMAL_STACK_SIZE];
static StaticTask_t dp_thread; static StackType_t dp_thread_stack[2 * configMINIMAL_STACK_SIZE];
static StaticTask_t isp_thread; static StackType_t isp_thread_stack[2 * configMINIMAL_STACK_SIZE];
static StaticTask_t uart_thread; static StackType_t uart_thread_stack[2 * configMINIMAL_STACK_SIZE];
static StaticTask_t uart_rx_thread; static StackType_t uart_rx_thread_stack[2 * configMINIMAL_STACK_SIZE];

static SemaphoreHandle_t isp_sem; static StaticSemaphore_t isp_sem_buffer;

#if HAS_ROTATION_SUPPORT == 1
static GFXMMU_HandleTypeDef hgfxmmu; static nema_cmdlist_t cl;
#endif

/* --- 5. HELPER FUNCTIONS --- */
static int is_cache_enable() {
#if defined(USE_DCACHE)
  return 1;
#else
  return 0;
#endif
}

static void decode_ld_landmark(roi_t *roi, ld_point_t *lm, ld_point_t *decoded)
{
  /* Calculate X and Y coordinates (Screen Projection) */
  decoded->x = roi->cx + (lm->x - 0.5) * roi->w * cos(roi->rotation) - (lm->y - 0.5) * roi->h * sin(roi->rotation);
  decoded->y = roi->cy + (lm->x - 0.5) * roi->w * sin(roi->rotation) + (lm->y - 0.5) * roi->h * cos(roi->rotation);

  /* Calculate Z coordinate (Depth Scaling) */
  /* We scale the normalized Z by the width of the bounding box to maintain aspect ratio with X/Y */
  decoded->z = lm->z * roi->w;
}

/* Helper Functions */
static void dp_update_drawing_area() {
  __disable_irq();
  SCRL_SetAddress_NoReload(lcd_fg_buffer[lcd_fg_buffer_rd_idx], SCRL_LAYER_1);
  __enable_irq();
}

static void dp_commit_drawing_area() {
  __disable_irq();
  SCRL_ReloadLayer(SCRL_LAYER_1);
  __enable_irq();
  lcd_fg_buffer_rd_idx = 1 - lcd_fg_buffer_rd_idx;
}

static void on_ld_toggle_button_click(void *args) {
  display_t *d = (display_t *) args;
  if (xSemaphoreTake(d->lock, portMAX_DELAY) == pdTRUE) {
    d->info.is_ld_displayed = !d->info.is_ld_displayed;
    xSemaphoreGive(d->lock);
  }
}

static void on_pd_toggle_button_click(void *args) {
  display_t *d = (display_t *) args;
  if (xSemaphoreTake(d->lock, portMAX_DELAY) == pdTRUE) {
    d->info.is_pd_displayed = !d->info.is_pd_displayed;
    xSemaphoreGive(d->lock);
  }
}

/* --- 6a. UART RECEIVE TASK (Laptop → Board) --- */
/*
 * Protocol:
 *   Sign mode : bare A-Z byte  (single char, from uart.py)
 *   STT mode  : 0x02 text 0x03 (STX/ETX framed, from stt.py)
 *
 *   STX (0x02) = start of new sentence — always resets the RX buffer.
 *   ETX (0x03) = end of sentence — finalizes and refreshes the display.
 *   If a stale accumulation exists, a new STX resets it cleanly.
 */
static void uart_receive_task(void *arg)
{
  uint8_t rx_byte;
  uint32_t timeout_count = 0;
  const uint32_t CLEAR_AFTER = 15;  /* clear after 15*100ms = 1.5s of no data */

  /* STT accumulation state */
  enum { RX_IDLE, RX_STT_ACCUM } state = RX_IDLE;
  char stt_buf[STT_TEXT_MAX];
  int  stt_idx = 0;
  uint32_t stt_timeout = 0;  /* safety: abort if no ETX for too long */

  while (1) {
    if (HAL_UART_Receive(&huart1, &rx_byte, 1, 100) == HAL_OK) {
      timeout_count = 0;

      /* ---- STX (0x02): always start a new sentence ---- */
      if (rx_byte == 0x02) {
        state = RX_STT_ACCUM;
        stt_idx = 0;
        stt_timeout = 0;
        continue;   /* nothing else to do for this byte */
      }

      if (state == RX_STT_ACCUM) {
        stt_timeout = 0;

        if (rx_byte == 0x03) {
          /* ---- ETX (0x03): sentence complete ---- */
          stt_buf[stt_idx] = '\0';

          /* Truncate to max 15 words */
          int word_count = 0;
          int trunc_len = 0;
          for (int i = 0; i < stt_idx; i++) {
            if (stt_buf[i] == ' ') {
              word_count++;
              if (word_count >= 15) { trunc_len = i; break; }
            }
          }
          if (trunc_len == 0) trunc_len = stt_idx;  /* fewer than 15 words */
          stt_buf[trunc_len] = '\0';

          /* Copy to global display buffer */
          for (int i = 0; i <= trunc_len; i++) stt_text[i] = stt_buf[i];
          stt_text_len = trunc_len;
          stt_active = 1;
          rx_char = '\0';  /* clear sign display */

          state = RX_IDLE;
          stt_idx = 0;
        } else if (rx_byte >= 0x20 && rx_byte < 0x7F) {
          /* Printable ASCII — append to buffer */
          if (stt_idx < STT_TEXT_MAX - 1) {
            stt_buf[stt_idx++] = (char)rx_byte;
          }
        }
        /* else: ignore non-printable bytes (e.g. \r, \n) */
      } else {
        /* ---- RX_IDLE: not accumulating STT ---- */
        if (rx_byte >= 'A' && rx_byte <= 'Z') {
          /* Sign prediction — clears STT display */
          rx_char = (char)rx_byte;
          stt_active = 0;
        }
        /* else: stray byte — ignore */
      }
    } else {
      /* Timeout (100 ms, no byte received) */
      timeout_count++;

      if (state == RX_STT_ACCUM) {
        stt_timeout++;
        if (stt_timeout >= 50) {  /* 5s with no ETX — abort accumulation */
          state = RX_IDLE;
          stt_idx = 0;
        }
      }

      if (state == RX_IDLE && !stt_active && timeout_count >= CLEAR_AFTER) {
        rx_char = '\0';
      }
    }
  }
}

/* --- 6b. UART LANDMARK TASK (Board → Laptop) --- */
static void uart_landmark_task(void *arg)
{
  char msg[1536]; /* Increased buffer size for added Z data */

  while (1) {
    vTaskDelay(pdMS_TO_TICKS(1000));

    if (xSemaphoreTake(disp.lock, portMAX_DELAY) == pdTRUE) {
       if (disp.info.pd_hand_nb > 0 && disp.info.hands[0].is_valid) {
           int offset = snprintf(msg, sizeof(msg), "(x,y,z) LM:");

           for (int k = 0; k < LD_LANDMARK_NB; k++) {
               ld_point_t decoded;
               decode_ld_landmark(&disp.info.hands[0].roi, &disp.info.hands[0].ld_landmarks[k], &decoded);

               /* UPDATED FORMAT: X, Y, Z */
               offset += snprintf(msg + offset, sizeof(msg) - offset, "%d,%d,%d,   ", (int)decoded.x, (int)decoded.y, (int)decoded.z);
           }
           snprintf(msg + offset - 1, sizeof(msg) - offset + 1, "\r\n");
           HAL_UART_Transmit(&huart1, (uint8_t*)msg, strlen(msg), HAL_MAX_DELAY);
       }
       xSemaphoreGive(disp.lock);
    }
  }
}

/* --- 7. LOGIC FUNCTIONS --- */
static float pd_normalize_angle(float angle) { return angle - 2 * M_PI * floorf((angle - (-M_PI)) / (2 * M_PI)); }
static float pd_cook_rotation(float angle) { return angle; }

static float pd_compute_rotation(pd_pp_box_t *box)
{
  float x0 = box->pKps[0].x; float y0 = box->pKps[0].y;
  float x1 = box->pKps[2].x; float y1 = box->pKps[2].y;
  float rotation = M_PI * 0.5 - atan2f(-(y1 - y0), x1 - x0);
  return pd_cook_rotation(pd_normalize_angle(rotation));
}

static void cvt_pd_coord_to_screen_coord(pd_pp_box_t *box)
{
  box->x_center *= LCD_BG_WIDTH; box->y_center *= LCD_BG_WIDTH;
  box->width *= LCD_BG_WIDTH; box->height *= LCD_BG_WIDTH;
  for (int i = 0; i < AI_PD_MODEL_PP_NB_KEYPOINTS; i++) {
    box->pKps[i].x *= LCD_BG_WIDTH; box->pKps[i].y *= LCD_BG_WIDTH;
  }
}

static void roi_shift_and_scale(roi_t *roi, float shift_x, float shift_y, float scale_x, float scale_y)
{
  float sx = (roi->w * shift_x * cos(roi->rotation) - roi->h * shift_y * sin(roi->rotation));
  float sy = (roi->w * shift_x * sin(roi->rotation) + roi->h * shift_y * cos(roi->rotation));
  roi->cx += sx; roi->cy += sy;
  float long_side = MAX(roi->w, roi->h);
  roi->w = long_side * scale_x; roi->h = long_side * scale_y;
}

static void pd_box_to_roi(pd_pp_box_t *box,  roi_t *roi)
{
  roi->cx = box->x_center; roi->cy = box->y_center;
  roi->w = box->width; roi->h = box->height;
  roi->rotation = pd_compute_rotation(box);
  roi_shift_and_scale(roi, 0, -0.5, 2.6, 2.6);
}

static void copy_pd_box(pd_pp_box_t *dst, pd_pp_box_t *src)
{
  dst->prob = src->prob; dst->x_center = src->x_center; dst->y_center = src->y_center;
  dst->width = src->width; dst->height = src->height;
  for (int i = 0 ; i < AI_PD_MODEL_PP_NB_KEYPOINTS; i++) dst->pKps[i] = src->pKps[i];
}

static void button_init(button_t *b, Button_TypeDef id, void (*on_click_handler)(void *), void *cb_args)
{
  BSP_PB_Init(id, BUTTON_MODE_GPIO);
  b->button_id = id; b->on_click_handler = on_click_handler;
  b->prev_state = 0; b->cb_args = cb_args;
}

static void button_process(button_t *b)
{
  int state = BSP_PB_GetState(b->button_id);
  if (state != b->prev_state && state && b->on_click_handler) b->on_click_handler(b->cb_args);
  b->prev_state = state;
}

static void cpuload_init(cpuload_info_t *cpu_load) { memset(cpu_load, 0, sizeof(cpuload_info_t)); }

static void cpuload_update(cpuload_info_t *cpu_load)
{
  cpu_load->history[1] = cpu_load->history[0];
  cpu_load->history[0].total = portGET_RUN_TIME_COUNTER_VALUE();
  cpu_load->history[0].thread = cpu_load->history[0].total - ulTaskGetIdleRunTimeCounter();
  cpu_load->history[0].tick = HAL_GetTick();
}

static void cpuload_get_info(cpuload_info_t *cpu_load, float *cpu_load_last, float *cpu_load_last_second, float *cpu_load_last_five_seconds)
{
  if (cpu_load_last) *cpu_load_last = 100.0 * (cpu_load->history[0].thread - cpu_load->history[1].thread) / (cpu_load->history[0].total - cpu_load->history[1].total);
}

static int bqueue_init(bqueue_t *bq, int buffer_nb, uint8_t **buffers)
{
  bq->free = xSemaphoreCreateCountingStatic(buffer_nb, buffer_nb, &bq->free_buffer);
  bq->ready = xSemaphoreCreateCountingStatic(buffer_nb, 0, &bq->ready_buffer);
  bq->buffer_nb = buffer_nb;
  for (int i = 0; i < buffer_nb; i++) bq->buffers[i] = buffers[i];
  bq->free_idx = bq->ready_idx = 0;
  return 0;
}

static uint8_t *bqueue_get_free(bqueue_t *bq, int is_blocking)
{
  if (xSemaphoreTake(bq->free, is_blocking ? portMAX_DELAY : 0) == pdFALSE) return NULL;
  uint8_t *res = bq->buffers[bq->free_idx];
  bq->free_idx = (bq->free_idx + 1) % bq->buffer_nb;
  return res;
}

static void bqueue_put_free(bqueue_t *bq) { xSemaphoreGive(bq->free); }

static uint8_t *bqueue_get_ready(bqueue_t *bq)
{
  xSemaphoreTake(bq->ready, portMAX_DELAY);
  uint8_t *res = bq->buffers[bq->ready_idx];
  bq->ready_idx = (bq->ready_idx + 1) % bq->buffer_nb;
  return res;
}

static void bqueue_put_ready(bqueue_t *bq)
{
  BaseType_t xHigherPriorityTaskWoken = pdFALSE;
  if (xPortIsInsideInterrupt()) { xSemaphoreGiveFromISR(bq->ready, &xHigherPriorityTaskWoken); portYIELD_FROM_ISR(xHigherPriorityTaskWoken); }
  else xSemaphoreGive(bq->ready);
}

static void reload_bg_layer(int next_disp_idx)
{
  SCRL_SetAddress_NoReload(lcd_bg_buffer[next_disp_idx], SCRL_LAYER_0);
  SCRL_ReloadLayer(SCRL_LAYER_0);
  SRCL_Update();
}

static void app_main_pipe_frame_event()
{
  int next_disp_idx = (lcd_bg_buffer_disp_idx + 1) % DISPLAY_BUFFER_NB;
  int next_capt_idx = (lcd_bg_buffer_capt_idx + 1) % DISPLAY_BUFFER_NB;
  HAL_DCMIPP_PIPE_SetMemoryAddress(CMW_CAMERA_GetDCMIPPHandle(), DCMIPP_PIPE1, DCMIPP_MEMORY_ADDRESS_0, (uint32_t) lcd_bg_buffer[next_capt_idx]);
  reload_bg_layer(next_disp_idx);
  lcd_bg_buffer_disp_idx = next_disp_idx; lcd_bg_buffer_capt_idx = next_capt_idx;
  frame_event_nb++;
}

static void app_ancillary_pipe_frame_event()
{
  uint8_t *next_buffer = bqueue_get_free(&nn_input_queue, 0);
  if (next_buffer) {
    HAL_DCMIPP_PIPE_SetMemoryAddress(CMW_CAMERA_GetDCMIPPHandle(), DCMIPP_PIPE2, DCMIPP_MEMORY_ADDRESS_0, (uint32_t) next_buffer);
    frame_event_nb_for_resize = frame_event_nb - 1;
    bqueue_put_ready(&nn_input_queue);
  }
}

static void app_main_pipe_vsync_event()
{
  BaseType_t xHigherPriorityTaskWoken = pdFALSE;
  if (xSemaphoreGiveFromISR(isp_sem, &xHigherPriorityTaskWoken) == pdTRUE) portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}

static int clamp_point(int *x, int *y)
{
  int xi = *x; int yi = *y;
  if (*x < 0) *x = 0;
  if (*y < 0) *y = 0;
  if (*x >= lcd_bg_area.XSize) *x = lcd_bg_area.XSize - 1;
  if (*y >= lcd_bg_area.YSize) *y = lcd_bg_area.YSize - 1;
  return (xi != *x) || (yi != *y);
}

static int clamp_point_with_margin(int *x, int *y, int margin)
{
  int xi = *x; int yi = *y;
  if (*x < margin) *x = margin;
  if (*y < margin) *y = margin;
  if (*x >= lcd_bg_area.XSize - margin) *x = lcd_bg_area.XSize - margin - 1;
  if (*y >= lcd_bg_area.YSize - margin) *y = lcd_bg_area.YSize - margin - 1;
  return (xi != *x) || (yi != *y);
}

static void display_pd_hand(pd_pp_box_t *hand)
{
  int x0, y0, x1, y1;
  x0 = (int)hand->x_center - ((int)hand->width + 1) / 2;
  y0 = (int)hand->y_center - ((int)hand->height + 1) / 2;
  x1 = (int)hand->x_center + ((int)hand->width + 1) / 2;
  y1 = (int)hand->y_center + ((int)hand->height + 1) / 2;
  clamp_point(&x0, &y0); clamp_point(&x1, &y1);
  UTIL_LCD_DrawRect(x0, y0, x1 - x0, y1 - y0, UTIL_LCD_COLOR_GREEN);
  for (int i = 0; i < 7; i++) {
    uint32_t color = (i != 0 && i != 2) ? UTIL_LCD_COLOR_RED : UTIL_LCD_COLOR_BLUE;
    int px = (int)hand->pKps[i].x; int py = (int)hand->pKps[i].y;
    clamp_point(&px, &py); UTIL_LCD_FillCircle(px, py, 2, color);
  }
}

static void rotate_point(float pt[2], float rotation)
{
  float x = pt[0]; float y = pt[1];
  pt[0] = cos(rotation) * x - sin(rotation) * y;
  pt[1] = sin(rotation) * x + cos(rotation) * y;
}

static void roi_to_corners(roi_t *roi, float corners[4][2])
{
  const float corners_init[4][2] = {{-roi->w/2, -roi->h/2}, {roi->w/2, -roi->h/2}, {roi->w/2, roi->h/2}, {-roi->w/2, roi->h/2}};
  for (int i = 0; i < 4; i++) {
    memcpy(corners[i], corners_init[i], sizeof(float)*2);
    rotate_point(corners[i], roi->rotation);
    corners[i][0] += roi->cx; corners[i][1] += roi->cy;
  }
}

static int clamp_corners(float corners_in[4][2], int corners_out[4][2])
{
  int is_clamp = 0;
  for (int i = 0; i < 4; i++) {
    corners_out[i][0] = (int)corners_in[i][0]; corners_out[i][1] = (int)corners_in[i][1];
    is_clamp |= clamp_point(&corners_out[i][0], &corners_out[i][1]);
  }
  return is_clamp;
}

static void display_roi(roi_t *roi)
{
  float corners_f[4][2]; int corners[4][2];
  roi_to_corners(roi, corners_f);
  if (clamp_corners(corners_f, corners)) return;
  for (int i = 0; i < 4; i++)
    UTIL_LCD_DrawLine(corners[i][0], corners[i][1], corners[(i + 1) % 4][0], corners[(i + 1) % 4][1], UTIL_LCD_COLOR_RED);
}

static void display_ld_hand(hand_info_t *hand)
{
  int x[LD_LANDMARK_NB], y[LD_LANDMARK_NB], is_clamped[LD_LANDMARK_NB];
  ld_point_t decoded;
  for (int i = 0; i < LD_LANDMARK_NB; i++) {
    decode_ld_landmark(&hand->roi, &hand->ld_landmarks[i], &decoded);
    x[i] = (int)decoded.x; y[i] = (int)decoded.y;
    is_clamped[i] = clamp_point_with_margin(&x[i], &y[i], DISK_RADIUS);
    if (!is_clamped[i]) UTIL_LCD_FillCircle(x[i], y[i], DISK_RADIUS, UTIL_LCD_COLOR_YELLOW);
  }
  for (int i = 0; i < LD_BINDING_NB; i++) {
    if (!is_clamped[ld_bindings_idx[i][0]] && !is_clamped[ld_bindings_idx[i][1]])
      UTIL_LCD_DrawLine(x[ld_bindings_idx[i][0]], y[ld_bindings_idx[i][0]], x[ld_bindings_idx[i][1]], y[ld_bindings_idx[i][1]], UTIL_LCD_COLOR_BLACK);
  }
}

void display_hand(display_info_t *info, hand_info_t *hand)
{
  if (info->is_pd_displayed) { display_pd_hand(&hand->pd_hands); display_roi(&hand->roi); }
  if (info->is_ld_displayed) display_ld_hand(hand);
}

static void Display_NetworkOutput(display_info_t *info)
{
  float cpu_load_one_second; int line_nb = 0;
  UTIL_LCD_FillRect(lcd_fg_area.X0, lcd_fg_area.Y0, lcd_fg_area.XSize, lcd_fg_area.YSize, 0x00000000);

  /* branding */
  UTIL_LCD_SetFont(&Font24);
  UTIL_LCD_SetTextColor(UTIL_LCD_COLOR_ST_BLUE_DARK); UTIL_LCDEx_PrintfAt(12, 12, LEFT_MODE, "SANCHAAR MITRA");
  UTIL_LCD_SetTextColor(0xFF00008B); UTIL_LCDEx_PrintfAt(10, 10, LEFT_MODE, "SANCHAAR MITRA");
  UTIL_LCD_DrawHLine(10, 42, 260, 0xFF00008B); UTIL_LCD_DrawHLine(10, 44, 260, 0xFF00008B);
  UTIL_LCD_SetFont(&LCD_FONT); UTIL_LCD_SetTextColor(UTIL_LCD_COLOR_RED);
  UTIL_LCDEx_PrintfAt(10, 55, LEFT_MODE, "ST Innovation Fair 2026"); UTIL_LCDEx_PrintfAt(10, 75, LEFT_MODE, "Team - JIIT");
  UTIL_LCD_SetTextColor(UTIL_LCD_COLOR_GREEN); UTIL_LCDEx_PrintfAt(0, LCD_FG_HEIGHT - 30, RIGHT_MODE, "By - Ankur Majumdar and Maitreya Agarwal");

  /* metrics */
  UTIL_LCD_SetTextColor(UTIL_LCD_COLOR_BLACK);
  cpuload_update(&cpu_load); cpuload_get_info(&cpu_load, &cpu_load_one_second, NULL, NULL);
  UTIL_LCDEx_PrintfAt(0, LINE(line_nb++),  RIGHT_MODE, "Cpu load");
  UTIL_LCDEx_PrintfAt(0, LINE(line_nb++),  RIGHT_MODE, "   %.1f%%", cpu_load_one_second);
  line_nb++;
  UTIL_LCDEx_PrintfAt(0, LINE(line_nb++), RIGHT_MODE, "Inferences");
  UTIL_LCDEx_PrintfAt(0, LINE(line_nb++), RIGHT_MODE, " pd %2ums", info->pd_ms);
  UTIL_LCDEx_PrintfAt(0, LINE(line_nb++), RIGHT_MODE, " hl %2ums", info->hl_ms);
  line_nb++;
  UTIL_LCDEx_PrintfAt(0, LINE(line_nb++), RIGHT_MODE, "  %.1f FPS", 1000.0 / info->nn_period_ms);

  for (int i = 0; i < info->pd_hand_nb; i++) if (info->hands[i].is_valid) display_hand(info, &info->hands[i]);

  /* Display received data */
  if (stt_active && stt_text_len > 0) {
    /* --- STT mode: show text higher on screen, word-wrapped, max 15 words --- */
    UTIL_LCD_SetFont(&LCD_FONT);
    UTIL_LCD_SetTextColor(UTIL_LCD_COLOR_CYAN);
    int stt_y_start = LCD_FG_HEIGHT / 2 - 30;
    UTIL_LCDEx_PrintfAt(10, stt_y_start, LEFT_MODE, "Voice:");

    UTIL_LCD_SetTextColor(UTIL_LCD_COLOR_WHITE);
    char line_buf[48];
    int chars_per_line = (LCD_FG_WIDTH - 20) / LCD_FONT.Width;
    if (chars_per_line > (int)sizeof(line_buf) - 1) chars_per_line = sizeof(line_buf) - 1;
    int y_pos = stt_y_start + LCD_FONT.Height + 4;
    int src = 0;
    int max_lines = 3;
    for (int ln = 0; ln < max_lines && src < stt_text_len; ln++) {
      int copy_len = stt_text_len - src;
      if (copy_len > chars_per_line) copy_len = chars_per_line;
      /* Break at last space if not at end */
      if (src + copy_len < stt_text_len) {
        int brk = copy_len;
        while (brk > 0 && stt_text[src + brk] != ' ') brk--;
        if (brk > 0) copy_len = brk + 1;
      }
      for (int c = 0; c < copy_len; c++) line_buf[c] = stt_text[src + c];
      line_buf[copy_len] = '\0';
      UTIL_LCDEx_PrintfAt(10, y_pos, LEFT_MODE, "%s", line_buf);
      y_pos += LCD_FONT.Height + 2;
      src += copy_len;
    }
  } else if (rx_char != '\0') {
    /* --- Sign mode: show single character --- */
    UTIL_LCD_SetFont(&Font24);
    UTIL_LCD_SetTextColor(UTIL_LCD_COLOR_WHITE);
    UTIL_LCDEx_PrintfAt(10, LCD_FG_HEIGHT - 60, LEFT_MODE, "Sign:");
    UTIL_LCD_SetTextColor(UTIL_LCD_COLOR_YELLOW);
    UTIL_LCDEx_PrintfAt(100, LCD_FG_HEIGHT - 60, LEFT_MODE, "%c", rx_char);
  }
}

static void palm_detector_init(pd_model_info_t *info)
{
  const LL_Buffer_InfoTypeDef *nn_out = LL_ATON_Output_Buffers_Info_palm_detector();
  const LL_Buffer_InfoTypeDef *nn_in = LL_ATON_Input_Buffers_Info_palm_detector();
  info->nn_in_len = LL_Buffer_len(&nn_in[0]);
  info->prob_out = (float *) LL_Buffer_addr_start(&nn_out[0]);
  info->prob_out_len = LL_Buffer_len(&nn_out[0]);
  info->boxes_out = (float *) LL_Buffer_addr_start(&nn_out[1]);
  info->boxes_out_len = LL_Buffer_len(&nn_out[1]);
  app_postprocess_init(&info->static_param, &NN_Instance_palm_detector);
}

static int palm_detector_run(uint8_t *buffer, pd_model_info_t *info, uint32_t *pd_exec_time)
{
  uint32_t start = HAL_GetTick();
  LL_ATON_Set_User_Input_Buffer_palm_detector(0, buffer, info->nn_in_len);
  LL_ATON_RT_Main(&NN_Instance_palm_detector);
  app_postprocess_run((void * []){info->prob_out, info->boxes_out}, 2, &info->pd_out, &info->static_param);
  int hand_nb = MIN(info->pd_out.box_nb, PD_MAX_HAND_NB);
  for (int i = 0; i < hand_nb; i++) { cvt_pd_coord_to_screen_coord(&info->pd_out.pOutData[i]); pd_box_to_roi(&info->pd_out.pOutData[i], &rois[i]); }
  CACHE_OP(SCB_InvalidateDCache_by_Addr(info->prob_out, info->prob_out_len));
  CACHE_OP(SCB_InvalidateDCache_by_Addr(info->boxes_out, info->boxes_out_len));
  *pd_exec_time = HAL_GetTick() - start;
  return hand_nb;
}

static void hand_landmark_init(hl_model_info_t *info)
{
  const LL_Buffer_InfoTypeDef *nn_out = LL_ATON_Output_Buffers_Info_hand_landmark();
  const LL_Buffer_InfoTypeDef *nn_in = LL_ATON_Input_Buffers_Info_hand_landmark();
  info->nn_in = LL_Buffer_addr_start(&nn_in[0]); info->nn_in_len = LL_Buffer_len(&nn_in[0]);
  info->prob_out = (float *) LL_Buffer_addr_start(&nn_out[2]); info->prob_out_len = LL_Buffer_len(&nn_out[2]);
  info->landmarks_out = (float *) LL_Buffer_addr_start(&nn_out[3]); info->landmarks_out_len = LL_Buffer_len(&nn_out[3]);
}

#if HAS_ROTATION_SUPPORT == 0
static int hand_landmark_prepare_input(uint8_t *buffer, roi_t *roi, hl_model_info_t *info)
{
  /* Full implementation for NO rotation support */
  float corners_f[4][2]; int corners[4][2];
  roi_to_corners(roi, corners_f);
  int is_clamped = clamp_corners(corners_f, corners);

  uint8_t* out_data = info->nn_in;
  int width_out = LD_WIDTH, height_out = LD_HEIGHT;

  if (is_clamped) {
    memset(info->nn_in, 0, info->nn_in_len);
  }

  uint8_t* in_data = buffer + corners[0][1] * LCD_BG_WIDTH * DISPLAY_BPP + corners[0][0]* DISPLAY_BPP;
  int width_in = corners[2][0] - corners[0][0];
  int height_in = corners[2][1] - corners[0][1];

  IPL_resize_bilinear_iu8ou8_with_strides_RGB(in_data, out_data, LCD_BG_WIDTH * DISPLAY_BPP, LD_WIDTH * DISPLAY_BPP,
                                              width_in, height_in, width_out, height_out);
  return 0;
}
#else
static void app_transform(nema_matrix3x3_t t, app_v3_t v) {
  app_v3_t r;
  for (int i = 0; i < 3; i++) r[i] = t[i][0] * v[0] + t[i][1] * v[1] + t[i][2] * v[2];
  for (int i = 0; i < 3; i++) v[i] = r[i];
}

static int hand_landmark_prepare_input(uint8_t *buffer, roi_t *roi, hl_model_info_t *info)
{
  app_v3_t vertex[] = {{0,0,1}, {LCD_BG_WIDTH,0,1}, {LCD_BG_WIDTH,LCD_BG_HEIGHT,1}, {0,LCD_BG_HEIGHT,1}};
  GFXMMU_BuffersTypeDef buffers = {0}; buffers.Buf0Address = (uint32_t) info->nn_in;
  HAL_GFXMMU_ModifyBuffers(&hgfxmmu, &buffers);

  nema_bind_dst_tex(GFXMMU_VIRTUAL_BUFFER0_BASE, LD_WIDTH, LD_HEIGHT, NEMA_RGBA8888, -1);
  nema_set_clip(0, 0, LD_WIDTH, LD_HEIGHT); nema_clear(0);
  nema_bind_src_tex((uintptr_t) buffer, LCD_BG_WIDTH, LCD_BG_HEIGHT, NEMA_RGBA8888, -1, NEMA_FILTER_BL);
  nema_enable_tiling(1); nema_set_blend_blit(NEMA_BL_SRC);

  nema_matrix3x3_t t; nema_mat3x3_load_identity(t);
  nema_mat3x3_translate(t, -roi->cx, -roi->cy);
  nema_mat3x3_rotate(t, nema_rad_to_deg(-roi->rotation));
  nema_mat3x3_scale(t, LD_WIDTH / roi->w, LD_HEIGHT / roi->h);
  nema_mat3x3_translate(t, LD_WIDTH / 2, LD_HEIGHT / 2);

  for (int i = 0 ; i < 4; i++) app_transform(t, vertex[i]);
  nema_blit_quad_fit(vertex[0][0], vertex[0][1], vertex[1][0], vertex[1][1], vertex[2][0], vertex[2][1], vertex[3][0], vertex[3][1]);
  nema_cl_submit(&cl); nema_cl_wait(&cl); HAL_ICACHE_Invalidate();
  return 0;
}
#endif

static int hand_landmark_run(uint8_t *buffer, hl_model_info_t *info, roi_t *roi, ld_point_t ld_landmarks[LD_LANDMARK_NB])
{
  if (hand_landmark_prepare_input(buffer, roi, info)) return 0;
  CACHE_OP(SCB_CleanInvalidateDCache_by_Addr(info->nn_in, info->nn_in_len));
  LL_ATON_RT_Main(&NN_Instance_hand_landmark);
  int is_valid = ld_post_process(info->prob_out, info->landmarks_out, ld_landmarks);
  CACHE_OP(SCB_InvalidateDCache_by_Addr(info->prob_out, info->prob_out_len));
  CACHE_OP(SCB_InvalidateDCache_by_Addr(info->landmarks_out, info->landmarks_out_len));
  return is_valid;
}

static void app_rot_init(hl_model_info_t *info)
{
  nema_init(); hgfxmmu.Instance = GFXMMU; hgfxmmu.Init.BlockSize = GFXMMU_12BYTE_BLOCKS;
  HAL_GFXMMU_Init(&hgfxmmu); GFXMMU_PackingTypeDef packing = {.Buffer0Activation = ENABLE, .Buffer0Mode = GFXMMU_PACKING_MSB_REMOVE, .DefaultAlpha = 0xff};
  HAL_GFXMMU_ConfigPacking(&hgfxmmu, &packing); cl = nema_cl_create_sized(8192); nema_cl_bind_circular(&cl);
}

static void compute_next_roi(roi_t *src, ld_point_t lm_in[LD_LANDMARK_NB], roi_t *next, pd_pp_box_t *next_pd)
{
  ld_point_t lm[LD_LANDMARK_NB];
  for (int i = 0; i < LD_LANDMARK_NB; i++) decode_ld_landmark(src, &lm_in[i], &lm[i]);
  /* simplified ld_to_roi */
  float max_x = -10000, max_y = -10000, min_x = 10000, min_y = 10000;
  const int indices[] = {0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18};
  for (int i = 0; i < 12; i++) {
    max_x = MAX(max_x, lm[indices[i]].x); max_y = MAX(max_y, lm[indices[i]].y);
    min_x = MIN(min_x, lm[indices[i]].x); min_y = MIN(min_y, lm[indices[i]].y);
  }
  next->cx = (max_x + min_x) / 2; next->cy = (max_y + min_y) / 2;
  next->w = (max_x - min_x); next->h = (max_y - min_y);
  next_pd->x_center = next->cx; next_pd->y_center = next->cy;
  next_pd->width = next->w; next_pd->height = next->h;
  /* Simplified rotation */
  float x0 = lm[0].x, y0 = lm[0].y, x1 = lm[9].x, y1 = lm[9].y;
  next->rotation = pd_cook_rotation(pd_normalize_angle(M_PI * 0.5 - atan2f(-(y1 - y0), x1 - x0)));
  roi_shift_and_scale(next, 0, -0.1, 2.0, 2.0);
}

static void nn_thread_fct(void *arg)
{
  hl_model_info_t hl_info; pd_model_info_t pd_info;
  pd_pp_point_t box_next_keypoints[AI_PD_MODEL_PP_NB_KEYPOINTS];
  pd_pp_box_t box_next = {.pKps = box_next_keypoints};
  int is_tracking = 0; roi_t roi_next;
  uint32_t pd_ms, hl_ms, nn_period_start, nn_period_ms, nn_period_filtered_ms = 0;

  palm_detector_init(&pd_info); hand_landmark_init(&hl_info);
  #if HAS_ROTATION_SUPPORT == 1
  app_rot_init(&hl_info);
  #endif

  uint8_t *nn_pipe_dst = bqueue_get_free(&nn_input_queue, 0);
  CAM_NNPipe_Start(nn_pipe_dst, CMW_MODE_CONTINUOUS);
  nn_period_start = HAL_GetTick();

  while (1) {
    uint32_t now = HAL_GetTick();
    nn_period_ms = now - nn_period_start; nn_period_start = now;
    nn_period_filtered_ms = USE_FILTERED_TS ? (15 * nn_period_filtered_ms + nn_period_ms) / 16 : nn_period_ms;

    uint8_t *capture_buffer = bqueue_get_ready(&nn_input_queue);
    int idx_for_resize = frame_event_nb_for_resize % DISPLAY_BUFFER_NB;

    if (!is_tracking) {
      is_tracking = palm_detector_run(capture_buffer, &pd_info, &pd_ms);
      if (is_tracking) copy_pd_box(&box_next, &pd_info.pd_out.pOutData[0]);
    } else {
      rois[0] = roi_next; copy_pd_box(&box_next, &pd_info.pd_out.pOutData[0]); pd_ms = 0;
    }
    bqueue_put_free(&nn_input_queue);

    if (is_tracking) {
      hl_ms = HAL_GetTick();
      is_tracking = hand_landmark_run(lcd_bg_buffer[idx_for_resize], &hl_info, &rois[0], ld_landmarks[0]);
      CACHE_OP(SCB_InvalidateDCache_by_Addr(lcd_bg_buffer[idx_for_resize], LCD_BG_WIDTH*LCD_BG_HEIGHT*DISPLAY_BPP));
      if (is_tracking) compute_next_roi(&rois[0], ld_landmarks[0], &roi_next, &box_next);
      hl_ms = HAL_GetTick() - hl_ms;
    } else hl_ms = 0;

    xSemaphoreTake(disp.lock, portMAX_DELAY);
    disp.info.pd_ms = is_tracking ? 0 : pd_ms; disp.info.hl_ms = is_tracking ? hl_ms : 0;
    disp.info.nn_period_ms = nn_period_filtered_ms;
    disp.info.pd_hand_nb = is_tracking; disp.info.pd_max_prob = pd_info.pd_out.pOutData[0].prob;
    disp.info.hands[0].is_valid = is_tracking;
    copy_pd_box(&disp.info.hands[0].pd_hands, &box_next);
    disp.info.hands[0].roi = rois[0];
    for (int j = 0; j < LD_LANDMARK_NB; j++) disp.info.hands[0].ld_landmarks[j] = ld_landmarks[0][j];
    xSemaphoreGive(disp.lock); xSemaphoreGive(disp.update);
  }
}

static void dp_thread_fct(void *arg)
{
  button_t ld_btn, pd_btn;
  display_info_t info;
  #ifdef STM32N6570_DK_REV
  button_init(&ld_btn, BUTTON_USER1, on_ld_toggle_button_click, &disp);
  button_init(&pd_btn, BUTTON_TAMP, on_pd_toggle_button_click, &disp);
  #else
  button_init(&ld_btn, BUTTON_USER, on_ld_toggle_button_click, &disp);
  button_init(&pd_btn, BUTTON_USER, on_pd_toggle_button_click, &disp);
  #endif

  while (1) {
    xSemaphoreTake(disp.update, portMAX_DELAY);
    button_process(&ld_btn); button_process(&pd_btn);
    xSemaphoreTake(disp.lock, portMAX_DELAY); info = disp.info; xSemaphoreGive(disp.lock);
    dp_update_drawing_area(); Display_NetworkOutput(&info);
    SCB_CleanDCache_by_Addr(lcd_fg_buffer[lcd_fg_buffer_rd_idx], LCD_FG_WIDTH * LCD_FG_HEIGHT* 2);
    dp_commit_drawing_area();
  }
}

static void isp_thread_fct(void *arg)
{
  while (1) {
    xSemaphoreTake(isp_sem, portMAX_DELAY);
    CAM_IspUpdate();
  }
}

static void Display_init()
{
  SCRL_LayerConfig layers_config[2] = {
    {.origin={0,0}, .size={LCD_BG_WIDTH,LCD_BG_HEIGHT}, .format=SCRL_RGB888, .address=lcd_bg_buffer[lcd_bg_buffer_disp_idx]},
    {.origin={0,0}, .size={LCD_FG_WIDTH,LCD_FG_HEIGHT}, .format=SCRL_ARGB4444, .address=lcd_fg_buffer[1]}
  };
  #if HAS_ROTATION_SUPPORT == 1
  layers_config[0].format = SCRL_ARGB8888;
  #endif
  SCRL_ScreenConfig screen_config = {.size={LCD_BG_WIDTH,LCD_BG_HEIGHT}, .format=SCRL_RGB565, .address=screen_buffer, .fps=CAMERA_FPS};
  SCRL_Init((SCRL_LayerConfig *[2]){&layers_config[0], &layers_config[1]}, &screen_config);
  UTIL_LCD_SetLayer(SCRL_LAYER_1); UTIL_LCD_Clear(UTIL_LCD_COLOR_TRANSPARENT);
  UTIL_LCD_SetFont(&LCD_FONT); UTIL_LCD_SetTextColor(UTIL_LCD_COLOR_WHITE);
}

void app_run()
{
  printf("Init application\n");
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  memset(lcd_bg_buffer, 0, sizeof(lcd_bg_buffer)); memset(lcd_fg_buffer, 0, sizeof(lcd_fg_buffer));
  Display_init(); bqueue_init(&nn_input_queue, 2, (uint8_t *[2]){nn_input_buffers[0], nn_input_buffers[1]});
  cpuload_init(&cpu_load); CAM_Init();
  isp_sem = xSemaphoreCreateCountingStatic(1, 0, &isp_sem_buffer);
  disp.update = xSemaphoreCreateCountingStatic(1, 0, &disp.update_buffer);
  disp.lock = xSemaphoreCreateMutexStatic(&disp.lock_buffer);
  CAM_DisplayPipe_Start(lcd_bg_buffer[0], CMW_MODE_CONTINUOUS);

  xTaskCreateStatic(nn_thread_fct, "nn", configMINIMAL_STACK_SIZE * 2, NULL, FREERTOS_PRIORITY(1), nn_thread_stack, &nn_thread);
  xTaskCreateStatic(dp_thread_fct, "dp", configMINIMAL_STACK_SIZE * 2, NULL, FREERTOS_PRIORITY(-2), dp_thread_stack, &dp_thread);
  xTaskCreateStatic(isp_thread_fct, "isp", configMINIMAL_STACK_SIZE * 2, NULL, FREERTOS_PRIORITY(2), isp_thread_stack, &isp_thread);

  /* Start UART Tasks */
  xTaskCreateStatic(uart_landmark_task, "uart_lm", configMINIMAL_STACK_SIZE * 2, NULL, tskIDLE_PRIORITY + 1, uart_thread_stack, &uart_thread);
  xTaskCreateStatic(uart_receive_task, "uart_rx", configMINIMAL_STACK_SIZE * 2, NULL, tskIDLE_PRIORITY + 1, uart_rx_thread_stack, &uart_rx_thread);
}

int CMW_CAMERA_PIPE_FrameEventCallback(uint32_t p) { if (p == DCMIPP_PIPE1) app_main_pipe_frame_event(); else app_ancillary_pipe_frame_event(); return HAL_OK; }
int CMW_CAMERA_PIPE_VsyncEventCallback(uint32_t p) { if (p == DCMIPP_PIPE1) app_main_pipe_vsync_event(); return HAL_OK; }
