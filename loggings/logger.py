import logging


logging.basicConfig(
    level=logging.DEBUG, 
    format= '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
    logging.FileHandler("app.log"),
    logging.StreamHandler()
]
)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
