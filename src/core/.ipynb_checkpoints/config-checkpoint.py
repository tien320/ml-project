
from pathlib import Path


# Project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent
#    __file__: đường dẫn file config.py hiện tại.
# .resolve(): lấy đường dẫn tuyệt đối.
# .parent: thư mục cha của src/, tức thư mục gốc project.

# Data
DATA_DIR = ROOT_DIR / "data"
DATA_PATH = DATA_DIR / "house_predict.csv"
TARGET_COLUMN = "SalePrice"
# Ta không cần viết lại đường dẫn CSV trong load_data.py, train.py hay những file khác.
# Chỉ cần import:
# from src.config import DATA_PATH

# Model artifacts
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "linear_regression.joblib"
# Đây là vị trí lưu model sau khi training hoàn tất.
# Sau này khi chạy inference, chương trình sẽ đọc model từ vị trí này thay vì train lại từ đầu.

# Training configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42

LEARNING_RATE = 0.01
EPOCHS = 5000


# Preprocessing configuration
FEATURE_COLUMNS_TO_DROP = ["Id", TARGET_COLUMN]


def ensure_directories():
    """
    Create required directories if they do not exist.
    """

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
#     Tự động tạo thư mục artifacts nếu chưa tồn tại.
# - parents=True: tạo cả thư mục cha nếu cần.
# - exist_ok=True: không báo lỗi nếu thư mục đã tồn tại.
