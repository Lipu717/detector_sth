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

namespace {

    const int INPUT_WIDTH = 640;
    const int INPUT_HEIGHT = 640;
    const float CONFIDENCE_THRESHOLD = 0.25F;
    const float NMS_THRESHOLD = 0.45F;
    const int CAMERA_INDEX = 0;

    struct LetterboxResult {
        cv::Mat image;
        float scale;
        int padX;
        int padY;
    };

    struct Detection {
        int classId;
        float confidence;
        cv::Rect box;
    };

    std::string executableDirectory()
    {
        char path[MAX_PATH] = {};
        const DWORD length = GetModuleFileNameA(nullptr, path, MAX_PATH);
        if (length == 0 || length == MAX_PATH) {
            return ".";
        }

        const std::string fullPath(path, length);
        const std::size_t slash = fullPath.find_last_of("\\/");
        return slash == std::string::npos ? "." : fullPath.substr(0, slash);
    }

    std::string joinPath(const std::string& directory, const std::string& name)
    {
        if (directory.empty() || directory == ".") {
            return name;
        }
        return directory + "\\" + name;
    }

    std::wstring widen(const std::string& text)
    {
        if (text.empty()) {
            return std::wstring();
        }

        const int count = MultiByteToWideChar(
            CP_ACP, 0, text.c_str(), -1, nullptr, 0);
        if (count <= 0) {
            throw std::runtime_error("Failed to convert the model path to Unicode.");
        }

        std::vector<wchar_t> buffer(static_cast<std::size_t>(count));
        MultiByteToWideChar(
            CP_ACP, 0, text.c_str(), -1, buffer.data(), count);
        return std::wstring(buffer.data());
    }

    bool fileExists(const std::string& path)
    {
        const DWORD attributes = GetFileAttributesA(path.c_str());
        return attributes != INVALID_FILE_ATTRIBUTES &&
            (attributes & FILE_ATTRIBUTE_DIRECTORY) == 0;
    }

    std::vector<std::string> loadClassNames(const std::string& path)
    {
        std::ifstream file(path.c_str());
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open class-name file: " + path);
        }

        std::vector<std::string> names;
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (!line.empty()) {
                names.push_back(line);
            }
        }

        if (names.empty()) {
            throw std::runtime_error("The class-name file is empty: " + path);
        }
        return names;
    }

    LetterboxResult letterbox(const cv::Mat& source)
    {
        const float scale = std::min(
            static_cast<float>(INPUT_WIDTH) / source.cols,
            static_cast<float>(INPUT_HEIGHT) / source.rows);

        const int resizedWidth = static_cast<int>(std::round(source.cols * scale));
        const int resizedHeight = static_cast<int>(std::round(source.rows * scale));

        cv::Mat resized;
        cv::resize(source, resized, cv::Size(resizedWidth, resizedHeight));

        const int totalPadX = INPUT_WIDTH - resizedWidth;
        const int totalPadY = INPUT_HEIGHT - resizedHeight;
        const int left = totalPadX / 2;
        const int right = totalPadX - left;
        const int top = totalPadY / 2;
        const int bottom = totalPadY - top;

        cv::Mat padded;
        cv::copyMakeBorder(
            resized,
            padded,
            top,
            bottom,
            left,
            right,
            cv::BORDER_CONSTANT,
            cv::Scalar(114, 114, 114));

        LetterboxResult result;
        result.image = padded;
        result.scale = scale;
        result.padX = left;
        result.padY = top;
        return result;
    }

    std::vector<float> makeInputTensor(const cv::Mat& letterboxedImage)
    {
        cv::Mat rgb;
        cv::cvtColor(letterboxedImage, rgb, cv::COLOR_BGR2RGB);

        cv::Mat floatImage;
        rgb.convertTo(floatImage, CV_32FC3, 1.0 / 255.0);

        std::vector<cv::Mat> channels;
        cv::split(floatImage, channels);

        const std::size_t planeSize =
            static_cast<std::size_t>(INPUT_WIDTH) * INPUT_HEIGHT;
        std::vector<float> tensor(3 * planeSize);

        for (int channel = 0; channel < 3; ++channel) {
            if (!channels[channel].isContinuous()) {
                channels[channel] = channels[channel].clone();
            }
            std::memcpy(
                tensor.data() + static_cast<std::size_t>(channel) * planeSize,
                channels[channel].ptr<float>(),
                planeSize * sizeof(float));
        }
        return tensor;
    }

    cv::Mat predictionsAsRows(
        float* outputData,
        const std::vector<int64_t>& shape)
    {
        if (shape.size() != 3 || shape[0] != 1) {
            throw std::runtime_error("Unexpected YOLO output rank or batch size.");
        }

        const int dimension1 = static_cast<int>(shape[1]);
        const int dimension2 = static_cast<int>(shape[2]);
        if (dimension1 <= 0 || dimension2 <= 0) {
            throw std::runtime_error("YOLO returned an invalid output shape.");
        }

        cv::Mat raw(dimension1, dimension2, CV_32F, outputData);
        if (dimension1 < dimension2) {
            cv::Mat transposed;
            cv::transpose(raw, transposed);
            return transposed;
        }
        return raw.clone();
    }

    std::vector<Detection> detect(
        Ort::Session& session,
        const char* inputName,
        const char* outputName,
        const cv::Mat& frame,
        const std::vector<std::string>& classNames)
    {
        const LetterboxResult prepared = letterbox(frame);
        std::vector<float> inputData = makeInputTensor(prepared.image);

        const std::vector<int64_t> inputShape = {
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
        std::vector<Ort::Value> outputs = session.Run(
            Ort::RunOptions{ nullptr },
            inputNames,
            &inputTensor,
            1,
            outputNames,
            1);

        if (outputs.empty() || !outputs[0].IsTensor()) {
            throw std::runtime_error("ONNX Runtime returned no tensor output.");
        }

        Ort::TensorTypeAndShapeInfo outputInfo =
            outputs[0].GetTensorTypeAndShapeInfo();
        const std::vector<int64_t> outputShape = outputInfo.GetShape();
        float* outputData = outputs[0].GetTensorMutableData<float>();
        cv::Mat predictions = predictionsAsRows(outputData, outputShape);

        const int numberOfClasses = predictions.cols - 4;
        if (numberOfClasses != static_cast<int>(classNames.size())) {
            std::ostringstream message;
            message << "Model has " << numberOfClasses
                << " classes, but coco_classes.txt has "
                << classNames.size() << ".";
            throw std::runtime_error(message.str());
        }

        std::vector<cv::Rect> candidateBoxes;
        std::vector<float> candidateScores;
        std::vector<int> candidateClassIds;

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

            float x1 = (centerX - width * 0.5F - prepared.padX) /
                prepared.scale;
            float y1 = (centerY - height * 0.5F - prepared.padY) /
                prepared.scale;
            float x2 = (centerX + width * 0.5F - prepared.padX) /
                prepared.scale;
            float y2 = (centerY + height * 0.5F - prepared.padY) /
                prepared.scale;

            x1 = std::max(0.0F, std::min(x1, static_cast<float>(frame.cols - 1)));
            y1 = std::max(0.0F, std::min(y1, static_cast<float>(frame.rows - 1)));
            x2 = std::max(0.0F, std::min(x2, static_cast<float>(frame.cols)));
            y2 = std::max(0.0F, std::min(y2, static_cast<float>(frame.rows)));

            const int left = static_cast<int>(std::floor(x1));
            const int top = static_cast<int>(std::floor(y1));
            const int right = static_cast<int>(std::ceil(x2));
            const int bottom = static_cast<int>(std::ceil(y2));
            if (right <= left || bottom <= top) {
                continue;
            }

            candidateBoxes.push_back(
                cv::Rect(left, top, right - left, bottom - top));
            candidateScores.push_back(bestScore);
            candidateClassIds.push_back(bestClass);
        }

        std::map<int, std::vector<int> > indicesByClass;
        for (int i = 0; i < static_cast<int>(candidateClassIds.size()); ++i) {
            indicesByClass[candidateClassIds[i]].push_back(i);
        }

        std::vector<Detection> detections;
        for (std::map<int, std::vector<int> >::const_iterator group =
            indicesByClass.begin();
            group != indicesByClass.end(); ++group) {
            const std::vector<int>& globalIndices = group->second;
            std::vector<cv::Rect> classBoxes;
            std::vector<float> classScores;

            for (std::size_t i = 0; i < globalIndices.size(); ++i) {
                const int globalIndex = globalIndices[i];
                classBoxes.push_back(candidateBoxes[globalIndex]);
                classScores.push_back(candidateScores[globalIndex]);
            }

            std::vector<int> keptLocalIndices;
            cv::dnn::NMSBoxes(
                classBoxes,
                classScores,
                CONFIDENCE_THRESHOLD,
                NMS_THRESHOLD,
                keptLocalIndices);

            for (std::size_t i = 0; i < keptLocalIndices.size(); ++i) {
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

    cv::Scalar colorForClass(int classId)
    {
        return cv::Scalar(
            (37 * classId + 80) % 255,
            (17 * classId + 160) % 255,
            (29 * classId + 220) % 255);
    }

    void drawDetection(
        cv::Mat& frame,
        const Detection& detection,
        const std::vector<std::string>& classNames)
    {
        const cv::Scalar color = colorForClass(detection.classId);
        cv::rectangle(frame, detection.box, color, 2);

        std::ostringstream labelStream;
        labelStream << classNames[detection.classId] << " "
            << std::fixed << std::setprecision(2)
            << detection.confidence;
        const std::string label = labelStream.str();

        int baseline = 0;
        const cv::Size textSize = cv::getTextSize(
            label, cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &baseline);
        const int textX = detection.box.x;
        const int textY = std::max(textSize.height + 6, detection.box.y);

        cv::rectangle(
            frame,
            cv::Point(textX, textY - textSize.height - 6),
            cv::Point(textX + textSize.width + 6, textY + baseline),
            color,
            cv::FILLED);
        cv::putText(
            frame,
            label,
            cv::Point(textX + 3, textY - 3),
            cv::FONT_HERSHEY_SIMPLEX,
            0.55,
            cv::Scalar(0, 0, 0),
            1,
            cv::LINE_AA);
    }

}  // namespace

int main(int argc, char** argv)
{
    try {
        const std::string programDirectory = executableDirectory();
        const std::string modelPath =
            argc > 1 ? argv[1]
            : joinPath(programDirectory, "yolo11n_coco.onnx");
        const std::string classPath =
            argc > 2 ? argv[2]
            : joinPath(programDirectory, "coco_classes.txt");

        if (!fileExists(modelPath)) {
            std::cerr << "Model file not found:\n" << modelPath << std::endl;
            return 1;
        }
        if (!fileExists(classPath)) {
            std::cerr << "Class-name file not found:\n" << classPath
                << std::endl;
            return 1;
        }

        const std::vector<std::string> classNames =
            loadClassNames(classPath);

        std::cout << "Loading model with ONNX Runtime: "
            << modelPath << std::endl;
        Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "yolo11-coco");
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);
        const std::wstring wideModelPath = widen(modelPath);
        Ort::Session session(
            environment, wideModelPath.c_str(), sessionOptions);

        Ort::AllocatorWithDefaultOptions allocator;
        Ort::AllocatedStringPtr inputName =
            session.GetInputNameAllocated(0, allocator);
        Ort::AllocatedStringPtr outputName =
            session.GetOutputNameAllocated(0, allocator);

        cv::VideoCapture camera(CAMERA_INDEX, cv::CAP_DSHOW);
        if (!camera.isOpened()) {
            camera.open(CAMERA_INDEX);
        }
        if (!camera.isOpened()) {
            std::cerr << "Cannot open camera index " << CAMERA_INDEX
                << "." << std::endl;
            return 1;
        }

        camera.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        camera.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

        std::cout << "Camera started. Press Q or Esc to quit." << std::endl;
        cv::namedWindow("YOLO11 COCO - ONNX Runtime", cv::WINDOW_NORMAL);

        while (true) {
            cv::Mat frame;
            if (!camera.read(frame) || frame.empty()) {
                std::cerr << "Failed to read a camera frame." << std::endl;
                break;
            }

            const int64 startTicks = cv::getTickCount();
            const std::vector<Detection> detections = detect(
                session,
                inputName.get(),
                outputName.get(),
                frame,
                classNames);
            const double elapsedSeconds =
                (cv::getTickCount() - startTicks) / cv::getTickFrequency();
            const double fps = elapsedSeconds > 0.0
                ? 1.0 / elapsedSeconds
                : 0.0;

            for (std::size_t i = 0; i < detections.size(); ++i) {
                drawDetection(frame, detections[i], classNames);
            }

            std::ostringstream status;
            status << "FPS " << std::fixed << std::setprecision(1) << fps
                << " | Objects " << detections.size();
            cv::putText(
                frame,
                status.str(),
                cv::Point(12, 30),
                cv::FONT_HERSHEY_SIMPLEX,
                0.75,
                cv::Scalar(0, 255, 0),
                2,
                cv::LINE_AA);

            cv::imshow("YOLO11 COCO - ONNX Runtime", frame);
            const int key = cv::waitKey(1) & 0xFF;
            if (key == 27 || key == 'q' || key == 'Q') {
                break;
            }
        }

        camera.release();
        cv::destroyAllWindows();
        return 0;
    }
    catch (const Ort::Exception& error) {
        std::cerr << "ONNX Runtime error:\n"
            << error.what() << std::endl;
        return 1;
    }
    catch (const cv::Exception& error) {
        std::cerr << "OpenCV error:\n" << error.what() << std::endl;
        return 1;
    }
    catch (const std::exception& error) {
        std::cerr << "Error:\n" << error.what() << std::endl;
        return 1;
    }
}
