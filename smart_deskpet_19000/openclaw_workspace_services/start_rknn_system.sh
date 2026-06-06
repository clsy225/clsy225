#!/bin/bash
# 启动RKNN NPU摄像头检测系统

echo "=== RKNN NPU摄像头检测系统启动脚本 ==="
echo ""

# 检查并杀死占用端口的进程
echo "清理端口占用..."
sudo lsof -ti:8080 | xargs sudo kill -9 2>/dev/null || true
sleep 1

# 检查RKNN模型
echo "检查RKNN模型..."
if [ -f "/userdata/best_v2.rknn" ]; then
    echo "✓ 找到模型: /userdata/best_v2.rknn"
    ls -lh "/userdata/best_v2.rknn"
else
    echo "✗ 未找到模型: /userdata/best_v2.rknn"
    echo "请确保模型文件存在"
    exit 1
fi

# 检查摄像头
echo ""
echo "检查摄像头设备..."
if [ -e "/dev/video62" ]; then
    echo "✓ 找到摄像头设备: /dev/video62"
    echo "尝试切换到真实摄像头模式..."
    # 这里可以添加真实摄像头启动逻辑
    SCRIPT="rknn_camera_server.py"
else
    echo "⚠ 未找到摄像头设备 /dev/video62"
    echo "使用模拟摄像头模式"
    SCRIPT="rknn_camera_simulator.py"
fi

# 启动服务器
echo ""
echo "启动RKNN服务器..."
cd /home/linaro/.openclaw/workspace
nohup python3 "$SCRIPT" > rknn_system.log 2>&1 &

# 等待启动
echo "等待服务器启动..."
sleep 3

# 检查状态
echo ""
echo "=== 服务器状态 ==="
if curl -s http://localhost:8080/test > /dev/null 2>&1; then
    echo "✓ 服务器运行正常"
    curl -s http://localhost:8080/test | python3 -m json.tool 2>/dev/null || echo "状态获取成功"
else
    echo "✗ 服务器启动失败"
    echo "查看日志: tail -f /home/linaro/.openclaw/workspace/rknn_system.log"
    exit 1
fi

echo ""
echo "=== 访问信息 ==="
echo "主页面: http://localhost:8080"
echo "视频流: http://localhost:8080/video"
echo "状态API: http://localhost:8080/stats"
echo "测试API: http://localhost:8080/test"
echo ""
echo "日志文件: /home/linaro/.openclaw/workspace/rknn_system.log"
echo "停止命令: pkill -f \"python.*8080\""
echo ""
echo "系统已启动完成！"