#define WIN32_LEAN_AND_MEAN

#include "drawingGUI.h"

#include <windows.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define GUI_CANVAS_X 24
#define GUI_CANVAS_Y 54

#define GUI_PANEL_WIDTH 420

#define GUI_BUTTON_CLEAR 1001
#define GUI_BUTTON_PREDICT 1002

typedef struct {
    DrawingGUIConfig config;

    HWND hwnd;
    HWND clearButton;
    HWND predictButton;

    HFONT titleFont;
    HFONT normalFont;
    HFONT smallFont;
    HFONT largeFont;

    float *canvas;
    float *input;
    float *outputs;

    bool drawing;
    bool erasing;
    bool hasPrediction;

    float lastGridX;
    float lastGridY;

    int predictedClass;

    int canvasPixelWidth;
    int canvasPixelHeight;

    int panelX;

    int windowWidth;
    int windowHeight;
} DrawingGUIState;

static float clampf(float value, float min, float max)
{
    if(value < min) return min;
    if(value > max) return max;
    return value;
}

static int maxInt(int a, int b)
{
    return a > b ? a : b;
}

static int findMaxOutput(const float *outputs, int outputCount)
{
    if(outputCount <= 0) return -1;

    int maxIndex = 0;
    float maxValue = outputs[0];

    for(int i = 1; i < outputCount; i++){
        if(outputs[i] > maxValue){
            maxValue = outputs[i];
            maxIndex = i;
        }
    }

    return maxIndex;
}

static const char *getOutputLabel(DrawingGUIState *state, int index)
{
    static char numberBuffer[32];

    if(state->config.outputLabels != NULL && state->config.outputLabels[index] != NULL){
        return state->config.outputLabels[index];
    }

    snprintf(numberBuffer, sizeof(numberBuffer), "%d", index);

    return numberBuffer;
}

static bool pointInsideCanvas(DrawingGUIState *state, int x, int y)
{
    return
        x >= GUI_CANVAS_X &&
        y >= GUI_CANVAS_Y &&
        x < GUI_CANVAS_X + state->canvasPixelWidth &&
        y < GUI_CANVAS_Y + state->canvasPixelHeight;
}

static void clearCanvas(DrawingGUIState *state)
{
    int pixelCount = state->config.canvasWidth * state->config.canvasHeight;

    memset(state->canvas, 0, pixelCount * sizeof(float));
    memset(state->outputs, 0, state->config.outputCount * sizeof(float));

    state->predictedClass = -1;
    state->hasPrediction = false;
}

static void defaultPrepareInput(DrawingGUIState *state)
{
    int canvasPixelCount = state->config.canvasWidth * state->config.canvasHeight;

    if(state->config.inputCount != canvasPixelCount) return;

    memcpy(state->input, state->canvas, canvasPixelCount * sizeof(float));
}

static void predict(DrawingGUIState *state)
{
    if(state->config.predict == NULL) return;

    if(state->config.prepareInput != NULL){
        state->config.prepareInput(
            state->canvas,
            state->config.canvasWidth,
            state->config.canvasHeight,
            state->input,
            state->config.inputCount,
            state->config.userData
        );
    }
    else{
        defaultPrepareInput(state);
    }

    state->config.predict(
        state->input,
        state->config.inputCount,
        state->outputs,
        state->config.outputCount,
        state->config.userData
    );

    state->predictedClass = findMaxOutput(state->outputs, state->config.outputCount);
    state->hasPrediction = state->predictedClass >= 0;

    InvalidateRect(state->hwnd, NULL, FALSE);
}

static void paintPoint(DrawingGUIState *state, float gridX, float gridY, bool erase)
{
    float radius = state->config.brushRadius;

    int minX = (int)floorf(gridX - radius);
    int maxX = (int)ceilf(gridX + radius);

    int minY = (int)floorf(gridY - radius);
    int maxY = (int)ceilf(gridY + radius);

    for(int y = minY; y <= maxY; y++){
        if(y < 0 || y >= state->config.canvasHeight) continue;

        for(int x = minX; x <= maxX; x++){
            if(x < 0 || x >= state->config.canvasWidth) continue;

            float dx = ((float)x + 0.5f) - gridX;
            float dy = ((float)y + 0.5f) - gridY;

            float distance = sqrtf(dx * dx + dy * dy);

            if(distance > radius) continue;

            float strength = 1.0f - distance / radius;
            strength = clampf(strength, 0.0f, 1.0f);

            int index = y * state->config.canvasWidth + x;

            if(erase){
                state->canvas[index] *= 1.0f - strength;

                if(state->canvas[index] < 0.01f) state->canvas[index] = 0.0f;
            }
            else{
                if(strength > state->canvas[index]) state->canvas[index] = strength;
            }
        }
    }
}

static void paintLine(DrawingGUIState *state, float x0, float y0, float x1, float y1, bool erase)
{
    float dx = x1 - x0;
    float dy = y1 - y0;

    float distance = sqrtf(dx * dx + dy * dy);

    int steps = (int)ceilf(distance * 4.0f);

    if(steps < 1) steps = 1;

    for(int i = 0; i <= steps; i++){
        float t = (float)i / (float)steps;

        float x = x0 + dx * t;
        float y = y0 + dy * t;

        paintPoint(state, x, y, erase);
    }
}

static void drawTextAt(HDC dc, HFONT font, COLORREF color, int x, int y, const char *text)
{
    HFONT oldFont = (HFONT)SelectObject(dc, font);

    SetTextColor(dc, color);
    SetBkMode(dc, TRANSPARENT);

    TextOutA(dc, x, y, text, (int)strlen(text));

    SelectObject(dc, oldFont);
}

static void drawCanvas(DrawingGUIState *state, HDC dc)
{
    HBRUSH borderBrush = CreateSolidBrush(RGB(70, 76, 88));

    RECT border = {
        GUI_CANVAS_X - 2,
        GUI_CANVAS_Y - 2,
        GUI_CANVAS_X + state->canvasPixelWidth + 2,
        GUI_CANVAS_Y + state->canvasPixelHeight + 2
    };

    FillRect(dc, &border, borderBrush);

    DeleteObject(borderBrush);

    HBRUSH pixelBrush = (HBRUSH)GetStockObject(DC_BRUSH);

    for(int y = 0; y < state->config.canvasHeight; y++){
        for(int x = 0; x < state->config.canvasWidth; x++){
            float value = state->canvas[y * state->config.canvasWidth + x];

            value = clampf(value, 0.0f, 1.0f);

            int shade = (int)(value * 255.0f);

            SetDCBrushColor(dc, RGB(shade, shade, shade));

            int px = GUI_CANVAS_X + x * state->config.cellSize;
            int py = GUI_CANVAS_Y + y * state->config.cellSize;

            RECT pixelRect = {
                px,
                py,
                px + state->config.cellSize,
                py + state->config.cellSize
            };

            FillRect(dc, &pixelRect, pixelBrush);
        }
    }

    if(state->config.cellSize >= 8){
        HPEN gridPen = CreatePen(PS_SOLID, 1, RGB(28, 30, 35));
        HPEN oldPen = (HPEN)SelectObject(dc, gridPen);

        for(int x = 0; x <= state->config.canvasWidth; x++){
            int px = GUI_CANVAS_X + x * state->config.cellSize;

            MoveToEx(dc, px, GUI_CANVAS_Y, NULL);
            LineTo(dc, px, GUI_CANVAS_Y + state->canvasPixelHeight);
        }

        for(int y = 0; y <= state->config.canvasHeight; y++){
            int py = GUI_CANVAS_Y + y * state->config.cellSize;

            MoveToEx(dc, GUI_CANVAS_X, py, NULL);
            LineTo(dc, GUI_CANVAS_X + state->canvasPixelWidth, py);
        }

        SelectObject(dc, oldPen);

        DeleteObject(gridPen);
    }
}

static void drawOutputs(DrawingGUIState *state, HDC dc)
{
    drawTextAt(dc, state->normalFont, RGB(230, 232, 238), state->panelX, 56, "Prediction");

    if(state->hasPrediction){
        const char *prediction = getOutputLabel(state, state->predictedClass);

        drawTextAt(dc, state->largeFont, RGB(245, 245, 250), state->panelX + 10, 82, prediction);
    }
    else{
        drawTextAt(dc, state->largeFont, RGB(100, 105, 115), state->panelX + 10, 82, "-");
    }

    float minOutput = 0.0f;
    float maxOutput = 1.0f;

    if(state->hasPrediction && state->config.outputCount > 0){
        minOutput = state->outputs[0];
        maxOutput = state->outputs[0];

        for(int i = 1; i < state->config.outputCount; i++){
            if(state->outputs[i] < minOutput) minOutput = state->outputs[i];
            if(state->outputs[i] > maxOutput) maxOutput = state->outputs[i];
        }
    }

    float outputRange = maxOutput - minOutput;

    if(fabsf(outputRange) < 0.000001f) outputRange = 1.0f;

    int startY = 180;

    for(int i = 0; i < state->config.outputCount; i++){
        int y = startY + i * 30;

        const char *label = getOutputLabel(state, i);

        char text[128];

        snprintf(text, sizeof(text), "%s   %.5f", label, state->outputs[i]);

        COLORREF textColor = RGB(205, 209, 218);

        if(state->hasPrediction && i == state->predictedClass) textColor = RGB(110, 215, 140);

        drawTextAt(dc, state->smallFont, textColor, state->panelX, y, text);

        int barX = state->panelX + 140;
        int barWidth = 230;

        RECT backgroundRect = {
            barX,
            y + 2,
            barX + barWidth,
            y + 18
        };

        HBRUSH backgroundBrush = CreateSolidBrush(RGB(42, 46, 55));

        FillRect(dc, &backgroundRect, backgroundBrush);

        DeleteObject(backgroundBrush);

        float normalized = 0.0f;

        if(state->hasPrediction) normalized = (state->outputs[i] - minOutput) / outputRange;

        normalized = clampf(normalized, 0.0f, 1.0f);

        RECT fillRect = {
            barX,
            y + 2,
            barX + (int)(normalized * barWidth),
            y + 18
        };

        COLORREF barColor = RGB(86, 156, 214);

        if(state->hasPrediction && i == state->predictedClass) barColor = RGB(90, 190, 120);

        HBRUSH fillBrush = CreateSolidBrush(barColor);

        FillRect(dc, &fillRect, fillBrush);

        DeleteObject(fillBrush);
    }
}

static void drawWindow(DrawingGUIState *state, HWND hwnd)
{
    PAINTSTRUCT ps;

    HDC dc = BeginPaint(hwnd, &ps);

    RECT clientRect;

    GetClientRect(hwnd, &clientRect);

    HDC backDc = CreateCompatibleDC(dc);
    HBITMAP backBitmap = CreateCompatibleBitmap(dc, clientRect.right, clientRect.bottom);
    HBITMAP oldBitmap = (HBITMAP)SelectObject(backDc, backBitmap);

    HBRUSH background = CreateSolidBrush(RGB(20, 22, 27));

    FillRect(backDc, &clientRect, background);

    DeleteObject(background);

    drawTextAt(backDc, state->titleFont, RGB(235, 238, 245), GUI_CANVAS_X, 16, state->config.title);

    drawCanvas(state, backDc);
    drawOutputs(state, backDc);

    int hintY = GUI_CANVAS_Y + state->canvasPixelHeight + 12;

    drawTextAt(backDc, state->smallFont, RGB(160, 166, 178), GUI_CANVAS_X, hintY, "LMB: draw    RMB: erase");

    BitBlt(dc, 0, 0, clientRect.right, clientRect.bottom, backDc, 0, 0, SRCCOPY);

    SelectObject(backDc, oldBitmap);

    DeleteObject(backBitmap);
    DeleteDC(backDc);

    EndPaint(hwnd, &ps);
}

static LRESULT CALLBACK DrawingGUIWindowProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    DrawingGUIState *state = (DrawingGUIState *)GetWindowLongPtrA(hwnd, GWLP_USERDATA);

    if(message == WM_NCCREATE){
        CREATESTRUCTA *createData = (CREATESTRUCTA *)lParam;

        state = (DrawingGUIState *)createData->lpCreateParams;

        state->hwnd = hwnd;

        SetWindowLongPtrA(hwnd, GWLP_USERDATA, (LONG_PTR)state);

        return TRUE;
    }

    if(state == NULL) return DefWindowProcA(hwnd, message, wParam, lParam);

    switch(message){
        case WM_CREATE:{
            int buttonY = GUI_CANVAS_Y + state->canvasPixelHeight + 42;

            state->clearButton = CreateWindowA("BUTTON", "Clear", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON, GUI_CANVAS_X, buttonY, 120, 34, hwnd, (HMENU)(INT_PTR)GUI_BUTTON_CLEAR, GetModuleHandleA(NULL), NULL);
            state->predictButton = CreateWindowA("BUTTON", "Predict", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON, GUI_CANVAS_X + 132, buttonY, 120, 34, hwnd, (HMENU)(INT_PTR)GUI_BUTTON_PREDICT, GetModuleHandleA(NULL), NULL);

            SendMessageA(state->clearButton, WM_SETFONT, (WPARAM)state->normalFont, TRUE);
            SendMessageA(state->predictButton, WM_SETFONT, (WPARAM)state->normalFont, TRUE);

            return 0;
        }

        case WM_COMMAND:{
            int id = LOWORD(wParam);

            if(id == GUI_BUTTON_CLEAR){
                clearCanvas(state);
                InvalidateRect(hwnd, NULL, FALSE);

                return 0;
            }

            if(id == GUI_BUTTON_PREDICT){
                predict(state);

                return 0;
            }

            break;
        }

        case WM_LBUTTONDOWN:
        case WM_RBUTTONDOWN:{
            int mouseX = (int)(short)LOWORD(lParam);
            int mouseY = (int)(short)HIWORD(lParam);

            if(!pointInsideCanvas(state, mouseX, mouseY)) return 0;

            state->drawing = true;
            state->erasing = message == WM_RBUTTONDOWN;

            state->lastGridX = (float)(mouseX - GUI_CANVAS_X) / (float)state->config.cellSize;
            state->lastGridY = (float)(mouseY - GUI_CANVAS_Y) / (float)state->config.cellSize;

            SetCapture(hwnd);

            paintPoint(state, state->lastGridX, state->lastGridY, state->erasing);

            InvalidateRect(hwnd, NULL, FALSE);

            return 0;
        }

        case WM_MOUSEMOVE:{
            if(!state->drawing) return 0;

            int mouseX = (int)(short)LOWORD(lParam);
            int mouseY = (int)(short)HIWORD(lParam);

            float gridX = (float)(mouseX - GUI_CANVAS_X) / (float)state->config.cellSize;
            float gridY = (float)(mouseY - GUI_CANVAS_Y) / (float)state->config.cellSize;

            paintLine(state, state->lastGridX, state->lastGridY, gridX, gridY, state->erasing);

            state->lastGridX = gridX;
            state->lastGridY = gridY;

            InvalidateRect(hwnd, NULL, FALSE);

            return 0;
        }

        case WM_LBUTTONUP:
        case WM_RBUTTONUP:{
            if(!state->drawing) return 0;

            state->drawing = false;

            ReleaseCapture();

            if(state->config.autoPredict) predict(state);

            return 0;
        }

        case WM_CAPTURECHANGED:{
            state->drawing = false;

            return 0;
        }

        case WM_ERASEBKGND:{
            return 1;
        }

        case WM_PAINT:{
            drawWindow(state, hwnd);

            return 0;
        }

        case WM_CLOSE:{
            DestroyWindow(hwnd);

            return 0;
        }

        case WM_DESTROY:{
            PostQuitMessage(0);

            return 0;
        }
    }

    return DefWindowProcA(hwnd, message, wParam, lParam);
}

int DrawingGUI_Run(const DrawingGUIConfig *config)
{
    if(config == NULL) return 1;
    if(config->predict == NULL) return 1;
    if(config->canvasWidth <= 0) return 1;
    if(config->canvasHeight <= 0) return 1;
    if(config->inputCount <= 0) return 1;
    if(config->outputCount <= 0) return 1;

    DrawingGUIState state = {0};

    state.config = *config;

    if(state.config.cellSize <= 0) state.config.cellSize = 16;
    if(state.config.brushRadius <= 0.0f) state.config.brushRadius = 1.5f;
    if(state.config.title == NULL) state.config.title = "Drawing network";

    int canvasPixelCount = state.config.canvasWidth * state.config.canvasHeight;

    if(state.config.prepareInput == NULL && state.config.inputCount != canvasPixelCount) return 1;

    state.canvas = calloc(canvasPixelCount, sizeof(float));
    state.input = calloc(state.config.inputCount, sizeof(float));
    state.outputs = calloc(state.config.outputCount, sizeof(float));

    if(state.canvas == NULL || state.input == NULL || state.outputs == NULL){
        free(state.canvas);
        free(state.input);
        free(state.outputs);

        return 1;
    }

    state.predictedClass = -1;

    state.canvasPixelWidth = state.config.canvasWidth * state.config.cellSize;
    state.canvasPixelHeight = state.config.canvasHeight * state.config.cellSize;

    state.panelX = GUI_CANVAS_X + state.canvasPixelWidth + 40;

    int canvasBottom = GUI_CANVAS_Y + state.canvasPixelHeight + 110;
    int outputBottom = 180 + state.config.outputCount * 30 + 60;

    state.windowWidth = state.panelX + GUI_PANEL_WIDTH;
    state.windowHeight = maxInt(canvasBottom, outputBottom);

    state.titleFont = CreateFontA(24, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, "Segoe UI");
    state.normalFont = CreateFontA(18, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, "Segoe UI");
    state.smallFont = CreateFontA(15, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, "Segoe UI");
    state.largeFont = CreateFontA(72, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, ANTIALIASED_QUALITY, DEFAULT_PITCH | FF_DONTCARE, "Segoe UI");

    HINSTANCE instance = GetModuleHandleA(NULL);

    WNDCLASSA windowClass = {0};

    windowClass.lpfnWndProc = DrawingGUIWindowProc;
    windowClass.hInstance = instance;
    windowClass.lpszClassName = "SAI_DrawingGUI";
    windowClass.hCursor = LoadCursor(NULL, IDC_CROSS);
    windowClass.hbrBackground = NULL;

    WNDCLASSA existingClass = {0};

    if(!GetClassInfoA(instance, windowClass.lpszClassName, &existingClass)){
        if(!RegisterClassA(&windowClass)){
            free(state.canvas);
            free(state.input);
            free(state.outputs);

            DeleteObject(state.titleFont);
            DeleteObject(state.normalFont);
            DeleteObject(state.smallFont);
            DeleteObject(state.largeFont);

            return 1;
        }
    }

    DWORD windowStyle = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;

    RECT windowRect = {0, 0, state.windowWidth, state.windowHeight};

    AdjustWindowRect(&windowRect, windowStyle, FALSE);

    int actualWidth = windowRect.right - windowRect.left;
    int actualHeight = windowRect.bottom - windowRect.top;

    HWND hwnd = CreateWindowExA(0, windowClass.lpszClassName, state.config.title, windowStyle, CW_USEDEFAULT, CW_USEDEFAULT, actualWidth, actualHeight, NULL, NULL, instance, &state);

    if(hwnd == NULL){
        free(state.canvas);
        free(state.input);
        free(state.outputs);

        DeleteObject(state.titleFont);
        DeleteObject(state.normalFont);
        DeleteObject(state.smallFont);
        DeleteObject(state.largeFont);

        return 1;
    }

    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);

    MSG message;

    while(GetMessageA(&message, NULL, 0, 0) > 0){
        TranslateMessage(&message);
        DispatchMessageA(&message);
    }

    free(state.canvas);
    free(state.input);
    free(state.outputs);

    DeleteObject(state.titleFont);
    DeleteObject(state.normalFont);
    DeleteObject(state.smallFont);
    DeleteObject(state.largeFont);

    return 0;
}