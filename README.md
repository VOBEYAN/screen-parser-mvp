# 大屏图纸组件检测与组件库匹配 MVP

该项目把技术方案落成一个可运行原型，复用甲方 `ai-schema-view/schema-md/catalog.md` 作为组件库数据源。

## 已实现能力

- 输入高保真大屏图或草图图片。
- 优先支持 YOLO 检测权重；没有权重时自动使用 OpenCV fallback 生成候选 bbox。
- 基于 bbox 几何关系生成初步层次树。
- 只对同级组件计算重叠问题，避免把合法父子包含误判为重叠。
- 基于组件类型、颜色、比例和组件 catalog 做 Top-1 组件库匹配。
- 可选最终内容判别层：裁剪检测框后识别文字、表格/图表/指标卡形态；配置多模态大模型后用视觉大模型重排组件库候选。
- 输出结构化 JSON、证据图、Markdown 报告和 HTML 报告。
- 提供增强合成数据生成脚本、YOLO 训练入口和 Graph Transformer 二阶段训练脚本。
- 已基于合成数据训练出演示级 YOLO 权重和 Graph Transformer 权重。

## 项目结构

```text
screen-parser-mvp/
  app/
    detectors.py            # YOLO / OpenCV 检测
    hierarchy.py            # 第一阶段规则层次解析
    overlap.py              # 同级组件重叠检测
    matcher.py              # 组件库 Top-1 匹配
    component_library.py    # 解析甲方组件 catalog
    graph_transformer.py    # 第二阶段 GNN / Graph Transformer 骨架
    pipeline.py             # 端到端解析流程
    reporter.py             # JSON / 图片 / Markdown / HTML 报告
    server.py               # 轻量 HTTP 上传服务
  scripts/
    parse_image.py
    generate_synthetic_data.py
    train_yolo.py
    train_graph_transformer.py
```

## 命令行运行

```bash
cd /Users/bytedance/Desktop/数据可视化对接/screen-parser-mvp
python3 scripts/parse_image.py ../设计稿1.jpg
```

输出会生成在：

```text
screen-parser-mvp/artifacts/run_*/result.json
screen-parser-mvp/artifacts/run_*/evidence.png
screen-parser-mvp/artifacts/run_*/report.md
screen-parser-mvp/artifacts/run_*/report.html
```

## 启动 Web 服务

```bash
cd /Users/bytedance/Desktop/数据可视化对接/screen-parser-mvp
python3 -m app.server --port 8765
```

浏览器打开：

```text
http://127.0.0.1:8765/
```

## 当前模型产物

```text
models/yolo_screen_components_demo.pt
models/graph_transformer_demo.pt
```

当前 YOLO 权重基于 `data/synthetic-demo` 合成数据训练：

```text
训练集：96 张
验证集：24 张
最终验证 mAP50：0.984
```

Graph Transformer 权重基于同一合成数据的 `meta/*.json` 训练，训练目标包括节点类型、层级和 parentId。

## 使用 YOLO 权重

当训练好权重后，可以这样接入：

```bash
python3 scripts/parse_image.py ../设计稿1.jpg \
  --yolo-model models/yolo_screen_components_demo.pt \
  --graph-model models/graph_transformer_demo.pt

python3 -m app.server \
  --yolo-model models/yolo_screen_components_demo.pt \
  --graph-model models/graph_transformer_demo.pt
```

项目使用 Hybrid 检测策略：优先合并 YOLO 输出，同时保留 OpenCV fallback，避免演示级权重在真实设计稿上漏检。

## 启用多模态最终判别层

不配置 API key 时，`--multimodal-classifier` 会启用本地图形/文字特征规则，作为轻量兜底：

```bash
python3 -m app.server \
  --yolo-model models/yolo_screen_components_component.pt \
  --graph-model models/graph_transformer_composited.pt \
  --reference-library data/component-reference \
  --multimodal-classifier
```

配置多模态大模型后，最后一层会把检测框裁剪图、候选组件和组件库信息发给视觉大模型，让模型判断框内是标题、表格、柱状图、折线图、指标卡、地图等内容，再重排组件库候选：

```bash
export SCREEN_PARSER_VLM_API_KEY="your-api-key"
export SCREEN_PARSER_VLM_MODEL="gpt-4o-mini"
export SCREEN_PARSER_VLM_BASE_URL="https://api.openai.com/v1"

python3 -m app.server \
  --yolo-model models/yolo_screen_components_component.pt \
  --graph-model models/graph_transformer_composited.pt \
  --reference-library data/component-reference \
  --multimodal-classifier
```

## 生成合成数据

```bash
python3 scripts/generate_synthetic_data.py --count 120 --width 960 --height 540 --out data/synthetic-demo
python3 scripts/generate_synthetic_data.py --count 120 --sketch --out data/synthetic-sketch
```

增强版组件级合成数据，覆盖 91 个组件、标题位置变化、组件叠加组合、框内文字/图表形态提示：

```bash
python3 scripts/generate_composited_training_data.py \
  --out data/composited-screen-component-v2 \
  --label-mode component \
  --train-count 910 \
  --val-count 182 \
  --components-per-screen 10 \
  --layout-mode mixed \
  --title-placement-mode diverse \
  --overlay-rate 0.55 \
  --include-sketch \
  --clean
```

粗粒度检测数据可以这样生成：

```bash
python3 scripts/generate_composited_training_data.py \
  --out data/composited-screen-title-v2 \
  --label-mode coarse \
  --train-count 900 \
  --val-count 180 \
  --components-per-screen 12 \
  --layout-mode dense \
  --title-placement-mode diverse \
  --overlay-rate 0.45 \
  --include-sketch \
  --clean
```

生成内容包括：

```text
images/*.png
labels/*.txt
meta/*.json
data.yaml
classes.txt
```

## 训练 YOLO

```bash
python3 scripts/train_yolo.py --data data/synthetic-demo/data.yaml --epochs 20 --imgsz 640 --batch 8 --base-model yolov8n.pt
```

使用增强后的 91 类组件数据训练：

```bash
python3 scripts/train_yolo.py \
  --data data/composited-screen-component-v2/data.yaml \
  --epochs 50 \
  --imgsz 960 \
  --batch 4 \
  --base-model yolov8n.pt \
  --project runs/detect \
  --name yolo_screen_components_component_v2

cp runs/detect/yolo_screen_components_component_v2/weights/best.pt \
  models/yolo_screen_components_component_v2.pt
```

使用增强后的粗粒度数据训练：

```bash
python3 scripts/train_yolo.py \
  --data data/composited-screen-title-v2/data.yaml \
  --epochs 50 \
  --imgsz 960 \
  --batch 4 \
  --base-model yolov8n.pt \
  --project runs/detect \
  --name yolo_screen_components_title_v2

cp runs/detect/yolo_screen_components_title_v2/weights/best.pt \
  models/yolo_screen_components_title_v2.pt
```

训练完成后启动新权重版：

```bash
python3 -m app.server \
  --yolo-model models/yolo_screen_components_component_v2.pt \
  --graph-model models/graph_transformer_composited.pt \
  --reference-library data/component-reference \
  --multimodal-classifier
```

## 第二阶段 Graph Transformer

当前已经提供真实训练脚本：

```bash
python3 scripts/train_graph_transformer.py --data data/synthetic-demo --epochs 8 --out models/graph_transformer_demo.pt
```

当前推理链路已经支持通过 `--graph-model` 启用 Graph Transformer 层次解析。生产落地时可以继续扩展边特征、sibling、overlap 标注训练，进一步替换和增强第一阶段规则解析。

## 当前边界

- 当前 YOLO 和 Graph Transformer 是基于合成数据训练的演示级权重。
- 真实设计稿仍存在合成域到真实域的差距，所以项目保留 OpenCV fallback。
- Graph Transformer 已训练节点类型、层级和 parentId，尚未正式接管线上解析，需要继续补真实样本评估。
- 草图支持目前通过合成数据和同一检测入口预留，生产效果需要草图样本微调。
