import os
import cv2
import preprocessing as pre
import model
import inference 
import params as pr 

MODE = int(input("[INFO] 실행 모드 선택 (1: 훈련, 2: 추론) : "))
SHOW = True

def run_train():
    print("=" * 50)
    print("")
    dataset_csv_path = pre.generate_dataset()

    print("=" * 50)
    print("[INFO] Started model training.")
    xgb_model, X_train, y_train, X_val, y_val = model.train_model(dataset_csv_path)

    print("=" * 50)
    print("[INFO] Successfully trained model")
    model.evaluate(xgb_model, X_train, y_train, X_val, y_val)

def run_infer():
    if not pr.input_dir or not os.path.exists(pr.input_dir):
        raise FileNotFoundError(f"[INFO] Can't find path.")

    print("=" * 50)
    print("[INFO] Started sharpening new image.")
    out_path = inference.enhance_image_patchwise(
        input_path=pr.input_dir,
        patch=pr.patch,
        stride=pr.stride
    )
    print("=" * 50)
    print(f"[INFO] 보정 완료! : {out_path}")

    if SHOW:
        img_in = cv2.imread(pr.input_dir, cv2.IMREAD_COLOR)
        img_out = cv2.imread(out_path, cv2.IMREAD_COLOR)

        cv2.imshow("Original Image", img_in)
        cv2.imshow("Sharpened Image", img_out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    if MODE == 1:
        run_train()
    elif MODE == 2:
        run_infer()
    else:
        raise ValueError("[INFO] MODE 값은 1(훈련) 또는 2(추론)만 허용됩니다.")
