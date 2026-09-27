#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

namespace {

    // 输入张量尺寸与检测阈值
    const int INPUT_WIDTH = 640;
    const int INPUT_HEIGHT = 640;
    const float CONFIDENCE_THRESHOLD = 0.25F;
    const float NMS_THRESHOLD = 0.45F;
    const int CAMERA_INDEX = 0;

    // letterbox 结果：缩放后的图像 + 缩放比例 + 填充偏移
    struct LetterboxResult {
        Mat image;
        float scale;
        int padX;
        int padY;
    };

    // 单个检测框：类别 id、置信度、矩形框
    struct Detection {
        int classId;
        float confidence;
        Rect box;
    };

    // 获取可执行文件所在目录，用于找模型和类别文件
    string executableDirectory()
    {
        char path[MAX_PATH] = {};
        const DWORD length = GetModuleFileNameA(nullptr, path, MAX_PATH);
        if (length == 0 || length == MAX_PATH) {
            return ".";
        }

        const string fullPath(path, length);
        const size_t slash = fullPath.find_last_of("\\/");
        return slash == string::npos ? "." : fullPath.substr(0, slash);
    }

    // 拼接目录和文件名
    string joinPath(const string& directory, const string& name)
    {
        if (directory.empty() || directory == ".") {
            return name;
        }
        return directory + "\\" + name;
    }

    // 把 ANSI 字符串转成宽字符，ONNX Runtime 在 Windows 上需要宽路径
    wstring widen(const string& text)
    {
        if (text.empty()) {
            return wstring();
        }

        const int count = MultiByteToWideChar(
            CP_ACP, 0, text.c_str(), -1, nullptr, 0);
        if (count <= 0) {
            throw runtime_error("Failed to convert the model path to Unicode.");
        }

        vector<wchar_t> buffer(static_cast<size_t>(count));
        MultiByteToWideChar(
            CP_ACP, 0, text.c_str(), -1, buffer.data(), count);
        return wstring(buffer.data());
    }

    // 判断文件是否存在且不是目录
    bool fileExists(const string& path)
    {
        const DWORD attributes = GetFileAttributesA(path.c_str());
        return attributes != INVALID_FILE_ATTRIBUTES &&
            (attributes & FILE_ATTRIBUTE_DIRECTORY) == 0;
    }

    // 读取类别名文件，每行一个类别
    vector<string> loadClassNames(const string& path)
    {
        ifstream file(path.c_str());
        if (!file.is_open()) {
            throw runtime_error("Cannot open class-name file: " + path);
        }

        vector<string> names;
        string line;
        while (getline(file, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (!line.empty()) {
                names.push_back(line);
            }
        }

        if (names.empty()) {
            throw runtime_error("The class-name file is empty: " + path);
        }
        return names;
    }

    // letterbox：等比缩放图像到 640x640，四周用灰色填充
    // 这样输入到 YOLO 时不会拉伸变形，后续坐标也能按比例还原
    LetterboxResult letterbox(const Mat& source)
    {
        const float scale = min(
            static_cast<float>(INPUT_WIDTH) / source.cols,
            static_cast<float>(INPUT_HEIGHT) / source.rows);

        const int resizedWidth = static_cast<int>(round(source.cols * scale));
        const int resizedHeight = static_cast<int>(round(source.rows * scale));

        Mat resized;
        resize(source, resized, Size(resizedWidth, resizedHeight));

        const int totalPadX = INPUT_WIDTH - resizedWidth;
        const int totalPadY = INPUT_HEIGHT - resizedHeight;
        const int left = totalPadX / 2;
        const int right = totalPadX - left;
        const int top = totalPadY / 2;
        const int bottom = totalPadY - top;

        Mat padded;
        copyMakeBorder(
            resized,
            padded,
            top,
            bottom,
            left,
            right,
            BORDER_CONSTANT,
            Scalar(114, 114, 114));

        LetterboxResult result;
        result.image = padded;
        result.scale = scale;
        result.padX = left;
        result.padY = top;
        return result;
    }

    // 把 letterbox 后的 BGR 图像转成 YOLO 需要的 float 张量
    // 输出格式：NCHW，即 [1, 3, 640, 640]，值归一化到 0~1
    vector<float> makeInputTensor(const Mat& letterboxedImage)
    {
        Mat rgb;
        cvtColor(letterboxedImage, rgb, COLOR_BGR2RGB);

        Mat floatImage;
        rgb.convertTo(floatImage, CV_32FC3, 1.0 / 255.0);

        vector<Mat> channels;
        split(floatImage, channels);

        const size_t planeSize =
            static_cast<size_t>(INPUT_WIDTH) * INPUT_HEIGHT;
        vector<float> tensor(3 * planeSize);

        for (int channel = 0; channel < 3; ++channel) {
            if (!channels[channel].isContinuous()) {
                channels[channel] = channels[channel].clone();
            }
            memcpy(
                tensor.data() + static_cast<size_t>(channel) * planeSize,
                channels[channel].ptr<float>(),
                planeSize * sizeof(float));
        }
        return tensor;
    }

    // 把 YOLO 输出整理成“每行一个候选框”的矩阵
    // YOLO11 的输出通常是 [1, 4+nc, 8400]，需要判断哪一维是候选数并转置
    Mat predictionsAsRows(
        float* outputData,
        const vector<int64_t>& shape)
    {
        if (shape.size() != 3 || shape[0] != 1) {
            throw runtime_error("Unexpected YOLO output rank or batch size.");
        }

        const int dimension1 = static_cast<int>(shape[1]);
        const int dimension2 = static_cast<int>(shape[2]);
        if (dimension1 <= 0 || dimension2 <= 0) {
            throw runtime_error("YOLO returned an invalid output shape.");
        }

        Mat raw(dimension1, dimension2, CV_32F, outputData);
        if (dimension1 < dimension2) {
            Mat transposed;
            transpose(raw, transposed);
            return transposed;
        }
        return raw.clone();
    }

    // 核心检测函数：预处理 → 跑 ONNX → 解码输出 → 按类别做 NMS
    // 返回所有通过阈值和 NMS 的检测框，坐标已经映射回原始图像
    vector<Detection> detect(
        Ort::Session& session,
        const char* inputName,
        const char* outputName,
        const Mat& frame,
        const vector<string>& classNames)
    {
        const LetterboxResult prepared = letterbox(frame);
        vector<float> inputData = makeInputTensor(prepared.image);

        const vector<int64_t> inputShape = {
            1, 3, INPUT_HEIGHT, INPUT_WIDTH
        };
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo,
            inputData.data(),
            inputData.size(),
            inputShape.data(),
            inputShape.size());

        const char* inputNames[] = { inputName };
        const char* outputNames[] = { outputName };
        vector<Ort::Value> outputs = session.Run(
            Ort::RunOptions{ nullptr },
            inputNames,
            &inputTensor,
            1,
            outputNames,
            1);

        if (outputs.empty() || !outputs[0].IsTensor()) {
            throw runtime_error("ONNX Runtime returned no tensor output.");
        }

        Ort::TensorTypeAndShapeInfo outputInfo =
            outputs[0].GetTensorTypeAndShapeInfo();
        const vector<int64_t> outputShape = outputInfo.GetShape();
        float* outputData = outputs[0].GetTensorMutableData<float>();
        Mat predictions = predictionsAsRows(outputData, outputShape);

        const int numberOfClasses = predictions.cols - 4;
        if (numberOfClasses != static_cast<int>(classNames.size())) {
            ostringstream message;
            message << "Model has " << numberOfClasses
                << " classes, but coco_classes.txt has "
                << classNames.size() << ".";
            throw runtime_error(message.str());
        }

        vector<Rect> candidateBoxes;
        vector<float> candidateScores;
        vector<int> candidateClassIds;

        // 逐个候选框解码：找最大类别分数，超过阈值才保留
        for (int row = 0; row < predictions.rows; ++row) {
            const float* values = predictions.ptr<float>(row);

            int bestClass = 0;
            float bestScore = values[4];
            for (int classId = 1; classId < numberOfClasses; ++classId) {
                const float score = values[4 + classId];
                if (score > bestScore) {
                    bestScore = score;
                    bestClass = classId;
                }
            }

            if (bestScore < CONFIDENCE_THRESHOLD) {
                continue;
            }

            const float centerX = values[0];
            const float centerY = values[1];
            const float width = values[2];
            const float height = values[3];

            // 把 letterbox 坐标还原到原图坐标
            float x1 = (centerX - width * 0.5F - prepared.padX) /
                prepared.scale;
            float y1 = (centerY - height * 0.5F - prepared.padY) /
                prepared.scale;
            float x2 = (centerX + width * 0.5F - prepared.padX) /
                prepared.scale;
            float y2 = (centerY + height * 0.5F - prepared.padY) /
                prepared.scale;

            x1 = max(0.0F, min(x1, static_cast<float>(frame.cols - 1)));
            y1 = max(0.0F, min(y1, static_cast<float>(frame.rows - 1)));
            x2 = max(0.0F, min(x2, static_cast<float>(frame.cols)));
            y2 = max(0.0F, min(y2, static_cast<float>(frame.rows)));

            const int left = static_cast<int>(floor(x1));
            const int top = static_cast<int>(floor(y1));
            const int right = static_cast<int>(ceil(x2));
            const int bottom = static_cast<int>(ceil(y2));
            if (right <= left || bottom <= top) {
                continue;
            }

            candidateBoxes.push_back(
                Rect(left, top, right - left, bottom - top));
            candidateScores.push_back(bestScore);
            candidateClassIds.push_back(bestClass);
        }

        // 按类别分组，方便对每个类别单独做 NMS
        map<int, vector<int> > indicesByClass;
        for (int i = 0; i < static_cast<int>(candidateClassIds.size()); ++i) {
            indicesByClass[candidateClassIds[i]].push_back(i);
        }

        vector<Detection> detections;
        for (map<int, vector<int> >::const_iterator group =
            indicesByClass.begin();
            group != indicesByClass.end(); ++group) {
            const vector<int>& globalIndices = group->second;
            vector<Rect> classBoxes;
            vector<float> classScores;

            for (size_t i = 0; i < globalIndices.size(); ++i) {
                const int globalIndex = globalIndices[i];
                classBoxes.push_back(candidateBoxes[globalIndex]);
                classScores.push_back(candidateScores[globalIndex]);
            }

            vector<int> keptLocalIndices;
            dnn::NMSBoxes(
                classBoxes,
                classScores,
                CONFIDENCE_THRESHOLD,
                NMS_THRESHOLD,
                keptLocalIndices);

            for (size_t i = 0; i < keptLocalIndices.size(); ++i) {
                const int globalIndex = globalIndices[keptLocalIndices[i]];
                Detection detection;
                detection.classId = candidateClassIds[globalIndex];
                detection.confidence = candidateScores[globalIndex];
                detection.box = candidateBoxes[globalIndex];
                detections.push_back(detection);
            }
        }

        return detections;
    }

    // 根据类别 id 生成稳定的颜色，不同类别颜色不同
    Scalar colorForClass(int classId)
    {
        return Scalar(
            (37 * classId + 80) % 255,
            (17 * classId + 160) % 255,
            (29 * classId + 220) % 255);
    }

    // 在图像上画一个检测框：矩形 + 类别名 + 置信度
    void drawDetection(
        Mat& frame,
        const Detection& detection,
        const vector<string>& classNames)
    {
        const Scalar color = colorForClass(detection.classId);
        rectangle(frame, detection.box, color, 2);

        ostringstream labelStream;
        labelStream << classNames[detection.classId] << " "
            << fixed << setprecision(2)
            << detection.confidence;
        const string label = labelStream.str();

        int baseline = 0;
        const Size textSize = getTextSize(
            label, FONT_HERSHEY_SIMPLEX, 0.55, 1, &baseline);
        const int textX = detection.box.x;
        const int textY = max(textSize.height + 6, detection.box.y);

        rectangle(
            frame,
            Point(textX, textY - textSize.height - 6),
            Point(textX + textSize.width + 6, textY + baseline),
            color,
            FILLED);
        putText(
            frame,
            label,
            Point(textX + 3, textY - 3),
            FONT_HERSHEY_SIMPLEX,
            0.55,
            Scalar(0, 0, 0),
            1,
            LINE_AA);
    }

}  // namespace

int main(int argc, char** argv)
{
    try {
        // 默认从可执行文件旁边找模型和类别文件，命令行参数可覆盖
        const string programDirectory = executableDirectory();
        const string modelPath =
            argc > 1 ? argv[1]
            : joinPath(programDirectory, "yolo11n_coco.onnx");
        const string classPath =
            argc > 2 ? argv[2]
            : joinPath(programDirectory, "coco_classes.txt");

        if (!fileExists(modelPath)) {
            cerr << "Model file not found:\n" << modelPath << endl;
            return 1;
        }
        if (!fileExists(classPath)) {
            cerr << "Class-name file not found:\n" << classPath
                << endl;
            return 1;
        }

        const vector<string> classNames =
            loadClassNames(classPath);

        // 初始化 ONNX Runtime 并加载模型
        cout << "Loading model with ONNX Runtime: "
            << modelPath << endl;
        Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "yolo11-coco");
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);
        const wstring wideModelPath = widen(modelPath);
        Ort::Session session(
            environment, wideModelPath.c_str(), sessionOptions);

        Ort::AllocatorWithDefaultOptions allocator;
        Ort::AllocatedStringPtr inputName =
            session.GetInputNameAllocated(0, allocator);
        Ort::AllocatedStringPtr outputName =
            session.GetOutputNameAllocated(0, allocator);

        // 打开摄像头，优先用 DSHOW 后端，失败再退回默认
        VideoCapture camera(CAMERA_INDEX, CAP_DSHOW);
        if (!camera.isOpened()) {
            camera.open(CAMERA_INDEX);
        }
        if (!camera.isOpened()) {
            cerr << "Cannot open camera index " << CAMERA_INDEX
                << "." << endl;
            return 1;
        }

        camera.set(CAP_PROP_FRAME_WIDTH, 1280);
        camera.set(CAP_PROP_FRAME_HEIGHT, 720);

        cout << "Camera started. Press Q or Esc to quit." << endl;
        namedWindow("YOLO11 COCO - ONNX Runtime", WINDOW_NORMAL);

        // 主循环：读帧 → 检测 → 画框 → 显示 → 处理按键
        while (true) {
            Mat frame;
            if (!camera.read(frame) || frame.empty()) {
                cerr << "Failed to read a camera frame." << endl;
                break;
            }

            const int64 startTicks = getTickCount();
            const vector<Detection> detections = detect(
                session,
                inputName.get(),
                outputName.get(),
                frame,
                classNames);
            const double elapsedSeconds =
                (getTickCount() - startTicks) / getTickFrequency();
            const double fps = elapsedSeconds > 0.0
                ? 1.0 / elapsedSeconds
                : 0.0;

            for (size_t i = 0; i < detections.size(); ++i) {
                drawDetection(frame, detections[i], classNames);
            }

            // 左上角显示 FPS 和目标数量
            ostringstream status;
            status << "FPS " << fixed << setprecision(1) << fps
                << " | Objects " << detections.size();
            putText(
                frame,
                status.str(),
                Point(12, 30),
                FONT_HERSHEY_SIMPLEX,
                0.75,
                Scalar(0, 255, 0),
                2,
                LINE_AA);

            imshow("YOLO11 COCO - ONNX Runtime", frame);
            const int key = waitKey(1) & 0xFF;
            if (key == 27 || key == 'q' || key == 'Q') {
                break;
            }
        }

        camera.release();
        destroyAllWindows();
        return 0;
    }
    catch (const Ort::Exception& error) {
        cerr << "ONNX Runtime error:\n"
            << error.what() << endl;
        return 1;
    }
    catch (const Exception& error) {
        cerr << "OpenCV error:\n" << error.what() << endl;
        return 1;
    }
    catch (const exception& error) {
        cerr << "Error:\n" << error.what() << endl;
        return 1;
    }
}
