import argparse
from ultralytics import YOLO
from rknn.api import RKNN



DATASET_PATH = './rknn_model_zoo/datasets/COCO/coco_subset_20.txt'
DEFAULT_RKNN_PATH = './pt/yolov8.rknn'
DEFAULT_QUANT = True

def main():
    parser = argparse.ArgumentParser(description="Export YOLO model to specified format")
    parser.add_argument('--pt',type=str, help='Export model to ONNX format')
    parser.add_argument('--rknn_pth',type=str, help='path to save rknn model')
    parser.add_argument('--rockchip',type=str, help='rockchip platform: rk3588')

    args = parser.parse_args()

    onnx_path = args.pt.split('/')
    onnx_path = onnx_path[-1].split('.')
    model_path = "/pt/"+onnx_path[0]+".onnx"

    model = YOLO(args.pt)

    try:
        path = model.export(format='rknn')
        print(f"Model exported to: {path}")

    except Exception as e:
        print("Failed to export model!!!")
        print(e)
        exit(1)

    platform = args.rockchip
    do_quant = DEFAULT_QUANT
    output_path = args.rknn_pth
    rknn = RKNN(verbose=False)

    # Pre-process config
    print('--> Config rknn model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[
                    [255, 255, 255]], target_platform=platform)
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=model_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=do_quant, dataset=DATASET_PATH)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Release
    rknn.release()
    

if __name__ == "__main__":
    main()
    