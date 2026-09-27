# YOLO11 COCO 摄像头目标检测<br>
这是一个在 Windows 和 Visual Studio 环境中运行的实时目标检测项目。程序使用 OpenCV 读取摄像头、处理画面并绘制检测结果，使用 ONNX Runtime 加载和运行预训练的 YOLO11n 模型。<br>
该模型已经在 COCO 数据集上完成训练，无需自行准备数据集或重新训练，可以直接识别人、汽车、猫、狗、杯子、瓶子、手机、键盘、鼠标、书等 80 类常见物体。<br>
##主要功能<br>
- 调用电脑摄像头并实时读取画面<br>
- 使用预训练 YOLO11n 模型进行目标检测<br>
- 显示目标类别、置信度和检测框<br>
- 使用 NMS 去除重复检测框<br>
- 自动还原缩放和填充后的目标坐标<br>
- 在画面左上角显示推理帧率和目标数量<br>
- 按 `Q` 或 `Esc` 退出程序<br>

项目使用 ONNX Runtime 完成模型推理。OpenCV 负责摄像头读取、图像预处理、NMS 和结果显示，因此可以绕过 OpenCV 4.5.4 无法解析 YOLO11 ONNX 计算图的问题。<br>

## 项目文件
```Roxy_detector/
├── .github/
│   └── copilot-instructions.md
├── .vs/
│   └── Roxy_detector/
│       ├── CopilotIndices/
│       │   └── 17.14.1661.41761/
│       │       ├── CodeChunks.db
│       │       └── SemanticSymbols.db
│       ├── FileContentIndex/
│       │   ├── 04d7e8f4-2baf-4cb0-b818-a7b95405f8cc.vsidx
│       │   ├── 516914dc-b3c1-4828-bf48-42955b445b6f.vsidx
│       │   ├── 7f6336cf-2060-4618-9adc-644b98df5b8d.vsidx
│       │   ├── 9daf015a-5135-4393-b045-90bc51cc2fb0.vsidx
│       │   └── a120730c-7c68-4e54-a2f8-1353be133049.vsidx
│       ├── copilot-chat/
│       │   └── a58d16a4/
│       │       └── sessions/
│       └── v17/
│           ├── ipch/
│           │   └── AutoPCH/
│           ├── .suo
│           ├── Browse.VC.db
│           ├── DocumentLayout.backup.json
│           ├── DocumentLayout.json
│           └── Solution.VC.db
├── packages/
│   └── Microsoft.ML.OnnxRuntime.1.30.0/
│       ├── build/
│       │   ├── native/
│       │   │   ├── include/
│       │   │   ├── Microsoft.ML.OnnxRuntime.props
│       │   │   └── Microsoft.ML.OnnxRuntime.targets
│       │   ├── net9.0-android35.0/
│       │   │   └── Microsoft.ML.OnnxRuntime.targets
│       │   ├── net9.0-ios18.0/
│       │   │   └── Microsoft.ML.OnnxRuntime.targets
│       │   ├── net9.0-maccatalyst18.0/
│       │   │   └── _._
│       │   ├── netstandard2.0/
│       │   │   ├── Microsoft.ML.OnnxRuntime.props
│       │   │   └── Microsoft.ML.OnnxRuntime.targets
│       │   └── netstandard2.1/
│       │       ├── Microsoft.ML.OnnxRuntime.props
│       │       └── Microsoft.ML.OnnxRuntime.targets
│       ├── buildTransitive/
│       │   ├── net9.0-android35.0/
│       │   │   └── Microsoft.ML.OnnxRuntime.targets
│       │   ├── net9.0-ios18.0/
│       │   │   └── Microsoft.ML.OnnxRuntime.targets
│       │   └── net9.0-maccatalyst18.0/
│       │       └── _._
│       ├── runtimes/
│       │   ├── android/
│       │   │   └── native/
│       │   ├── ios/
│       │   │   └── native/
│       │   ├── linux-arm64/
│       │   │   └── native/
│       │   ├── linux-x64/
│       │   │   └── native/
│       │   ├── osx-arm64/
│       │   │   └── native/
│       │   ├── win-arm64/
│       │   │   └── native/
│       │   └── win-x64/
│       │       └── native/
│       ├── .signature.p7s
│       ├── LICENSE
│       ├── Microsoft.ML.OnnxRuntime.1.30.0.nupkg
│       ├── ORT_icon_for_light_bg.png
│       ├── Privacy.md
│       ├── README.md
│       └── ThirdPartyNotices.txt
├── x64/
│   └── Debug/
│       ├── Roxy_detector.exe
│       ├── Roxy_detector.pdb
│       ├── coco_classes.txt
│       ├── onnxruntime.dll
│       ├── onnxruntime_providers_shared.dll
│       └── yolo11n_coco.onnx
├── Roxy_detector.sln
├── Roxy_detector.vcxproj
├── Roxy_detector.vcxproj.filters
├── Roxy_detector.vcxproj.user
├── packages.config
└── roxy_detector_onnxruntime.cpp
```

## 安装 ONNX Runtime
在 Visual Studio 中右键项目，选择“管理 NuGet 程序包”，搜索并安装：<br>
```text
Microsoft.ML.OnnxRuntime 1.30.0
```
NuGet 会自动配置头文件、链接库，并在生成时复制 `onnxruntime.dll`。<br>

## 可识别类别示例
模型支持 COCO 数据集中的 80 类物体，常用类别包括：<br>
```text
person       人
car          汽车
bicycle      自行车
cat          猫
dog          狗
bottle       瓶子
cup          杯子
chair        椅子
laptop       笔记本电脑
mouse        鼠标
keyboard     键盘
cell phone   手机
book         书
clock        时钟
scissors     剪刀
teddy bear   玩具熊
```

## 模型说明

`yolo11n_coco.onnx` 是预训练的通用目标检测模型，适合验证摄像头读取、图像预处理、模型推理、输出解析和画框流程。它不能识别特定动漫角色；若需要识别洛琪希等特定目标，需要准备对应数据并进行微调或增加分类模型。<br>
