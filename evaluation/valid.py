from inference import inference
from evaluation.evaluate import evaluate


def valid(model, data_loader_test, pred_dir, method='isnet', testset='DIS-VD', only_S_MAE=True):
    model.eval()
    inference(model, data_loader_test, pred_dir, method, testset)
    performance_dict = evaluate(pred_dir, method, testset, only_S_MAE=only_S_MAE, epoch=model.epoch)
    return performance_dict
