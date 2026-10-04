#ifndef DRAWING_GUI_H
#define DRAWING_GUI_H

#include <stdbool.h>

typedef void (*DrawingGuiPredictCallback)(
    const float *input,
    int inputCount,
    float *outputs,
    int outputCount,
    void *userData
);

typedef void (*DrawingGuiPrepareInputCallback)(
    const float *canvas,
    int canvasWidth,
    int canvasHeight,
    float *input,
    int inputCount,
    void *userData
);

typedef struct {
    const char *title;

    int canvasWidth;
    int canvasHeight;
    int cellSize;

    int inputCount;
    int outputCount;

    float brushRadius;

    bool autoPredict;

    const char **outputLabels;

    DrawingGuiPrepareInputCallback prepareInput;
    DrawingGuiPredictCallback predict;

    void *userData;
} DrawingGUIConfig;

int DrawingGUI_Run(const DrawingGUIConfig *config);

#endif