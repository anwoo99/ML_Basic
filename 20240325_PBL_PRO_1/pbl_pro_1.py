import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import inspect
import time
import traceback
from torchvision import datasets, transforms
from itertools import product
from datetime import datetime, timedelta


"""
모델 Class 정의
"""
# MLP 모델 정의
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layer_depth, activation_func):
        super(MLP, self).__init__()
        layers = []

        for _ in range(layer_depth):
            layers.append(nn.Linear(input_size, hidden_size))
            if activation_func == 'relu':
                layers.append(nn.ReLU())
            elif activation_func == 'sigmoid':
                layers.append(nn.Sigmoid())
            input_size = hidden_size # 다음 input size 업데이트

        layers.append(nn.Linear(hidden_size, output_size))
        self.model = nn.Sequential(*layers) # 레이어 연결
            

    def forward(self, x):
        x = x.view(-1, 28 * 28) # Origin(BatchSize x 28 x 28) => Convert(BatchSize x 784)
        x = self.model(x)
        return x

class CNN(nn.Module):
    pass


"""
함수 정의
"""
def get_log_path(app_name):
    now = datetime.now()
    weekday = (now.weekday() + 1) % 7
    sys_path = "./log/"

    if not os.path.exists(sys_path):
        os.makedirs(sys_path)

    return os.path.join(sys_path, f"{app_name}-{weekday}.log")


def log(app_name, content):
    caller = inspect.currentframe().f_back
    caller_function = inspect.getframeinfo(caller).function
    log_path = get_log_path(app_name)
    date_head = datetime.now().strftime("%m/%d %H:%M:%S")

    try:
        modified_time = time.localtime(os.path.getmtime(log_path))
        modified_yday = modified_time.tm_yday
    except FileNotFoundError:
        modified_yday = -1

    mode = 'a+' if datetime.now().timetuple().tm_yday == modified_yday else 'w+'

    logmsg = f"[{date_head}] [{caller_function}] {content}\n"

    with open(log_path, mode) as fd:
        fd.write(logmsg)

def get_loader(batch_size):
    try:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # MNIST DataSet 가져오기(Train/Test)
        train_dataset = datasets.MNIST(
            root='./data/', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(
            root='./data/', train=False, transform=transform)

        # Dataloader: Mini batch를 만들어주는 역할을 함
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader
    except Exception as err:
        raise


def train(model, train_loader, criterion, optimizer):
    try:
        model.train()  # 훈련모드 설정
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()  # Gradient 초기화(중첩 방지)
            output = model(data)

            loss = criterion(output, target)  # 손실함수 적용
            loss.backward()  # 역전파 실행

            optimizer.step()  # 최적화 실행
            running_loss += loss.item()

        return running_loss / len(train_loader)
    except Exception as err:
        raise


def test(model, test_loader, criterion):
    try:
        model.eval()  # 평가모드 설정
        test_loss = 0.0
        correct = 0

        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                # 단순 평가이므로 역전파는 실행하지 않음
                test_loss += criterion(output, target).item()

                """
                - output(batch_size X 10(0~9))
                - 행 기준으로 가장 큰 값을 가지는 열의 인덱스를 선택하면, 그 값이 의미하는 바와 일치할 확률이 높음
                """
                pred = output.argmax(
                    dim=1, keepdim=True)  # output에서 가장 큰 값을 가지는 인덱스 선택(행 기준)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)  # 테스트 데이터셋 전체에 대한 평균 손실을 의미
        accuracy = 100. * correct / len(test_loader.dataset)
        return test_loss, accuracy
    except Exception as err:
        raise


def Main():
    try:
        app_name = "PBL_PRO_1"

        """
        HyperParameter 설정
        """
        num_epochs = [10, 20, 30]
        batch_sizes = [64, 128, 256]
        learning_rates = [0.001, 0.0001]
        layer_depths = [2, 3, 4]
        hidden_sizes = [256, 512, 1024]
        activation_funcs = ['relu', 'sigmoid']
        criterion = nn.CrossEntropyLoss()
        optimizers = [optim.Adam, optim.SGD]
        models = ['mlp']

        best_performance = {'accuracy': 0, 'hyperparameters': None, 'model': None}

        total_start_time = time.time()  # 전체 러닝 타임 측정을 위한 시작 시간 기록

        log(app_name, "Start to grid search..!")

        # 모든 하이퍼파라미터 조합(product)에 대해 "그리드 서치" 실행
        for epoch, batch_size, lr, ld, hs, af, opt, model_type in product(num_epochs, batch_sizes, learning_rates, layer_depths, hidden_sizes, activation_funcs, optimizers, models):
            log(app_name, f"Training with Epochs: {epoch}, Batch Size: {batch_size}, Learning Rate: {lr}, Layer Depth: {ld}, Hidden Size: {hs}, Activation Function: {af}, Optimizer: {opt.__name__}, Model Type: {model_type}")

            if model_type == 'mlp':
                model = MLP(input_size=28*28, hidden_size=hs, output_size=10, layer_depth=ld, activation_func=af)
            elif model_type == 'cnn':
                continue
            else:
                continue
            
            # 모델 정보 출력
            log(app_name, model.summary())

            optimizer = opt(model.parameters(), lr=lr)
            train_loader, test_loader = get_loader(batch_size)

            epoch_start_time = time.time()  # 현재 실험의 시작 시간 기록

            for epoch_idx in range(1, epoch + 1):
                train_loss = train(model=model, train_loader=train_loader,
                                   criterion=criterion, optimizer=optimizer)
                test_loss, accuracy = test(model, test_loader, criterion)
                log(app_name, f"Epoch [{epoch_idx}/{epoch}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")

                # 가장 좋은 성능을 달성한 경우 기록
                if accuracy > best_performance['accuracy']:
                    best_performance['accuracy'] = accuracy
                    best_performance['hyperparameters'] = {'Epochs': epoch, 'Batch Size': batch_size, 'Learning Rate': lr, 'Layer Depth': ld, 'Hidden Size': hs, 'Activation Function': af, 'Optimizer': opt.__name__, 'Model Type': model_type}
                    best_performance['model'] = model

            epoch_end_time = time.time()  # 현재 실험의 종료 시간 기록
            elapsed_time = timedelta(seconds=epoch_end_time - epoch_start_time)  # 현재 실험의 러닝 타임 계산
            log(app_name, f"Elapsed Time for Experiment: {elapsed_time}")

        total_end_time = time.time()  # 전체 러닝 타임 측정을 위한 종료 시간 기록
        total_elapsed_time = timedelta(seconds=total_end_time - total_start_time)  # 전체 러닝 타임 계산
        log(app_name, f"Total Elapsed Time: {total_elapsed_time}")

        # 가장 좋은 성능을 달성한 하이퍼파라미터와 모델 정보 출력
        log(app_name, f"Best Performance: Accuracy: {best_performance['accuracy']:.2f}%, Hyperparameters: {best_performance['hyperparameters']}, Model: {best_performance['model']}")

    except Exception as err:
        traceback_error = traceback.format_exc()
        log(app_name, traceback_error)
        sys.exit()


if __name__ == "__main__":
    Main()
