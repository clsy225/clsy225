#!/bin/bash
# 启动完整的绫地宁宁系统（含声纹识别整合页）

echo "=== 启动绫地宁宁完整系统 ==="
echo "时间: $(date)"
echo ""

# 停止旧服务
echo "停止旧服务..."
pkill -f "python3.*(ningning|camera|soviet|integrated_browser|tts5_web)" 2>/dev/null || true
sleep 2

# 检查依赖
echo "检查Python依赖..."
python3 -c "import cv2" 2>/dev/null && echo "✓ OpenCV可用" || echo "⚠ OpenCV未安装"
python3 -c "import numpy" 2>/dev/null && echo "✓ NumPy可用" || echo "⚠ NumPy未安装"
python3 -c "import flask, requests" 2>/dev/null && echo "✓ Flask/Requests可用" || echo "⚠ Flask/Requests未安装"

# 检查TTS
echo "检查TTS..."
if [ -f "/home/linaro/tts_speak.sh" ]; then
    echo "✓ TTS脚本存在"
    chmod +x /home/linaro/tts_speak.sh
else
    echo "⚠ TTS脚本不存在，创建测试脚本"
    cat > /home/linaro/.openclaw/workspace/tts_test.sh << 'EOF'
#!/bin/bash
echo "测试TTS: $1"
sleep 1
echo "TTS完成"
EOF
    chmod +x /home/linaro/.openclaw/workspace/tts_test.sh
fi

# 创建日志目录
mkdir -p /home/linaro/.openclaw/logs

# 启动系统
echo ""
echo "启动绫地宁宁整合系统..."
cd /home/linaro/.openclaw/workspace
nohup ./start_tts5_web.sh >/dev/null 2>&1 &
sleep 2
nohup python3 integrated_browser.py > /home/linaro/.openclaw/logs/ningning_integrated_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 等待启动
sleep 3

# 检查状态
echo ""
echo "检查服务状态..."
if curl -s http://localhost:8083/ > /dev/null; then
    echo "✓ 系统启动成功!"
    echo ""
    echo "访问地址: http://localhost:8083"
    echo ""
    echo "现在已整合："
    echo "  1. 左侧：摄像头/TTS 控制台"
    echo "  2. 右侧：TTS5 声纹建库与识别"
    echo ""
    echo "测试命令:"
    echo "  curl -X POST http://localhost:8083/infer \\" 
    echo "    -H 'Content-Type: application/json' \\" 
    echo "    -d '{\"text\": \"绫地宁宁现在摄像头有几个人\"}'"
else
    echo "✗ 系统启动失败，检查日志..."
    tail -20 /home/linaro/.openclaw/logs/ningning_integrated_*.log 2>/dev/null || echo "无日志文件"
fi

echo ""
echo "=== 启动完成 ==="
