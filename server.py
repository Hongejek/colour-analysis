"""
图片颜色聚类可视化 — 后端服务器
功能：
  1. 提供静态文件服务（index.html、images/ 等）
  2. 代理 AI API 调用，保护 API Key 不暴露到前端

启动方式：
  python server.py

配置方式（按优先级）：
  1. 环境变量 AI_API_KEY      — AI API 密钥（必需）
  2. 环境变量 AI_API_ENDPOINT  — API 端点 URL（可选，默认 closeaiasia）
  3. 环境变量 PORT             — 服务端口（可选，默认 8080）
"""

import os
import json
import logging
from flask import Flask, request, jsonify, send_from_directory

# ---------- 配置 ----------
API_KEY = "sk-22a65671a941485e842d6d2fba567a88"
API_ENDPOINT = os.environ.get(
    "AI_API_ENDPOINT",
    "https://api.deepseek.com/v1/chat/completions"
)
PORT = int(os.environ.get("PORT", 8080))

# ---------- 日志 ----------
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger(__name__)

# ---------- Flask 应用 ----------
app = Flask(__name__, static_folder=".", static_url_path="")


@app.route("/")
def index():
    """首页"""
    return send_from_directory(".", "index.html")


@app.route("/images/<path:filename>")
def serve_images(filename):
    """图片静态资源"""
    return send_from_directory("images", filename)


# ==================== AI 分析 API ====================

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    接收前端发来的聚类颜色数据，调用 AI API 分析颜色和谐度。
    
    请求体 JSON：
    {
        "colors": [
            {"id": 0, "hex": "#78757d", "rgb": [120,117,125], "percentage": 29.1},
            ...
        ],
        "colorSpace": "RGB",
        "model": "gpt-4o"
    }
    
    返回 JSON：
    {
        "success": true,
        "content": "AI 分析结果文本..."
    }
    """
    if not API_KEY:
        return jsonify({
            "success": False,
            "error": "服务器未配置 AI_API_KEY 环境变量。请启动时设置：$env:AI_API_KEY='sk-...' (PowerShell) 或 export AI_API_KEY='sk-...' (Bash)"
        }), 500

    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"success": False, "error": "请求体必须是有效的 JSON"}), 400

    colors = data.get("colors", [])
    color_space = data.get("colorSpace", "RGB")
    model = "deepseek-v4-flash"  # 后端固定模型

    if not colors:
        return jsonify({"success": False, "error": "缺少 colors 数据"}), 400

    # 构建 Prompt
    prompt = _build_harmony_prompt(colors, color_space)

    # 调用 AI API
    try:
        ai_response = _call_ai_api(model, prompt)
        return jsonify({"success": True, "content": ai_response})
    except Exception as e:
        log.error(f"AI API 调用失败: {e}")
        return jsonify({"success": False, "error": str(e)}), 502


def _build_harmony_prompt(colors, color_space):
    """构建颜色和谐度分析的 Prompt"""
    sorted_colors = sorted(colors, key=lambda c: c.get("id", 0))
    color_list = "\n".join(
        f"- 簇{c['id']}: {c['hex']} "
        f"(R:{c['rgb'][0]} G:{c['rgb'][1]} B:{c['rgb'][2]}) "
        f"— 占比 {c['percentage']:.1f}%"
        for c in sorted_colors
    )

    return f"""你是一位专业的色彩设计与配色理论专家。请分析以下图片经过 K-Means 聚类（{color_space} 色彩空间）后得到的一组颜色均值，判断它们是否构成和谐的配色方案。

## 颜色列表（按聚类结果）：
{color_list}

## 分析要求：
1. **和谐度评分**（满分 100）：给出一个综合分数。
2. **色彩关系分析**：基于色彩理论（互补色、类似色、三角色、分裂互补等），分析这些颜色之间的关系。
3. **情感与氛围**：这组颜色传达什么样的视觉感受和情感氛围？
4. **改进建议**：如果和谐度不高，可以调整哪些颜色来提升和谐度？

请用中文回复，格式清晰，包含以上四个部分。开头直接给出评分，格式为："## 和谐度评分：XX/100" """


def _call_ai_api(model, prompt):
    """调用 OpenAI 兼容的 Chat Completions API"""
    import urllib.request
    import urllib.error

    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一位专业的色彩设计与配色理论专家。请用中文回复。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 2048,
    }).encode("utf-8")

    req = urllib.request.Request(
        API_ENDPOINT,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not content:
                raise RuntimeError("AI 未返回有效内容")
            return content
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        try:
            err_json = json.loads(error_body)
            msg = err_json.get("error", {}).get("message", error_body)
        except Exception:
            msg = error_body[:500]
        raise RuntimeError(f"AI API 返回错误 (HTTP {e.code}): {msg}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"无法连接到 AI API 服务器: {e.reason}")


# ==================== 启动 ====================

if __name__ == "__main__":
    if not API_KEY:
        log.warning("⚠️  未设置 AI_API_KEY 环境变量，AI 分析功能将不可用。")
        log.warning("   请设置: $env:AI_API_KEY='sk-...'   (PowerShell)")
        log.warning("   或:     export AI_API_KEY='sk-...' (Bash)")
    else:
        log.info(f"✅ AI API Key 已配置 (密钥长度: {len(API_KEY)} 字符)")
    log.info(f"📡 AI API 端点: {API_ENDPOINT}")
    log.info(f"🌐 服务启动: http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
