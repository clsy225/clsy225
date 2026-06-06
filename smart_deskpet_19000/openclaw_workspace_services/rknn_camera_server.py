    global stop_camera
    print("停止摄像头...")
    stop_camera = True
    time.sleep(1)
    print("清理完成")

if __name__ == '__main__':
    try:
        # 启动摄像头线程
        camera_thread = threading.Thread(target=camera_worker, daemon=True)
        camera_thread.start()
        
        time.sleep(2)
        
        print("=" * 60)
        print("RKNN摄像头服务器")
        print("端口: 8086")
        print("摄像头: /dev/video62")
        print(f"模型: {current_model}")
        print("功能: 多模型切换 | 置信度条 | 实时统计")
        print("=" * 60)
        
        app.run(host='0.0.0.0', port=8086, threaded=True, debug=False)
    except KeyboardInterrupt:
        print("\n中断")
    finally:
        cleanup()