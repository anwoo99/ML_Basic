import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import sys
import os
import inspect
import time
import traceback
import math
from torchvision import datasets, transforms
from itertools import product
from datetime import datetime, timedelta

APP_NAME = "PBL_PRO_1"

"""
모델 Class 정의
"""
# MLP 모델 정의
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layer_depth, activation_func, dropout_rate, weight_init):
        super(MLP, self).__init__()
        self.activation_func = self._get_activation_func(activation_func)
        layers = []

        for _ in range(layer_depth):
            layers.extend([
                nn.Linear(input_size, hidden_size),
                self.activation_func,
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_size)
            ])
            input_size = hidden_size

        layers.append(nn.Linear(hidden_size, output_size))
        self.model = nn.Sequential(*layers)
        self._initialize_weights(weight_init)

    def _get_activation_func(self, activation_func):
        activation_functions = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU(),
            'selu': nn.SELU(),
            'softmax': nn.Softmax(dim=-1)
        }
        func = activation_functions.get(activation_func)
        if func is None:
            raise ValueError(f"Invalid activation function: {activation_func}")
        return func

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.model(x)
        return x

    def _initialize_weights(self, weight_init):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if weight_init == 'uniform':
                    init.uniform_(m.weight, a=-0.5, b=0.5)
                elif weight_init == 'normal':
                    init.normal_(m.weight, mean=0, std=0.01)
                elif weight_init == 'xavier':
                    init.xavier_uniform_(m.weight)
                elif weight_init == 'he':
                    init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                elif weight_init == 'kaiming':
                    init.kaiming_uniform_(m.weight)


class CNN(nn.Module):
    def __init__(self, output_size, activation_func, dropout_rate, weight_init):
        super(CNN, self).__init__()
        self.activation_func = self._get_activation_func(activation_func)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        self._initialize_weights(weight_init)

    def forward(self, x):
        x = self.activation_func(self.conv1(x))
        x = nn.MaxPool2d(2)(x)
        x = self.activation_func(self.conv2(x))
        x = nn.MaxPool2d(2)(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.activation_func(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def _get_activation_func(self, activation_func):
        activation_functions = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU(),
            'selu': nn.SELU(),
            'softmax': nn.Softmax(dim=-1)
        }
        func = activation_functions.get(activation_func)
        if func is None:
            raise ValueError(f"Invalid activation function: {activation_func}")
        return func

    def _initialize_weights(self, weight_init):
        if weight_init == 'uniform':
            init.uniform_(self.conv1.weight)
            init.uniform_(self.conv2.weight)
            init.uniform_(self.fc1.weight)
            init.uniform_(self.fc2.weight)
        elif weight_init == 'normal':
            init.normal_(self.conv1.weight, mean=0, std=0.01)
            init.normal_(self.conv2.weight, mean=0, std=0.01)
            init.normal_(self.fc1.weight, mean=0, std=0.01)
            init.normal_(self.fc2.weight, mean=0, std=0.01)
        elif weight_init == 'xavier':
            init.xavier_uniform_(self.conv1.weight)
            init.xavier_uniform_(self.conv2.weight)
            init.xavier_uniform_(self.fc1.weight)
            init.xavier_uniform_(self.fc2.weight)
        elif weight_init == 'he':
            init.kaiming_uniform_(self.conv1.weight, a=math.sqrt(5))
            init.kaiming_uniform_(self.conv2.weight, a=math.sqrt(5))
            init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
            init.kaiming_uniform_(self.fc2.weight, a=math.sqrt(5))
        elif weight_init == 'kaiming':
            init.kaiming_uniform_(self.conv1.weight)
            init.kaiming_uniform_(self.conv2.weight)
            init.kaiming_uniform_(self.fc1.weight)
            init.kaiming_uniform_(self.fc2.weight)


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
            transforms.RandomRotation(degrees=10), # Data Augementation(각도 10으로 돌리기)
            transforms.Normalize((0.5,), (0.5,))
        ])

        # MNIST DataSet 가져오기(Train/Test)
        train_dataset = datasets.MNIST(
            root='./data/', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(
            root='./data/', train=False, transform=transform)

        # Dataloader: Mini batch를 만들어주는 역할을 함
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

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


def run_experiment(ii, epoch, batch_size, lr, ld, hs, af, drr, weight_decay, weight_init, opt, model_type, criterion, best_performance):
    log(APP_NAME, f"[SEQ({ii:03d})] Training with Epochs: {epoch}, Batch Size: {batch_size}, Learning Rate: {lr}, Layer Depth: {ld}, Hidden Size: {hs}, Activation Function: {af}, Dropout Rate: {drr}, Weight Decay: {weight_decay}, Weight Initialization: {weight_init}, Optimizer: {opt.__name__}, Model Type: {model_type}")

    if model_type == 'mlp':
        model = MLP(input_size=28*28, hidden_size=hs, output_size=10, layer_depth=ld, activation_func=af, dropout_rate=drr, weight_init=weight_init)
    elif model_type == 'cnn':
        model = CNN(output_size=10, activation_func=af, dropout_rate=drr, weight_init=weight_init)
    else:
        return

    optimizer = opt(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader, test_loader = get_loader(batch_size)

    epoch_start_time = time.time()

    for epoch_idx in range(1, epoch + 1):
        train_loss = train(model=model, train_loader=train_loader, criterion=criterion, optimizer=optimizer)
        test_loss, accuracy = test(model=model, test_loader=test_loader, criterion=criterion)

        log(APP_NAME, f"Epoch [{epoch_idx}/{epoch}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")

        if accuracy > best_performance['accuracy']:
            best_performance['accuracy'] = accuracy
            best_performance['hyperparameters'] = {'Epochs': epoch, 'Batch Size': batch_size, 'Learning Rate': lr, 'Layer Depth': ld, 'Hidden Size': hs, 'Activation Function': af, 'Dropout Rate': drr, 'Weight Decay': weight_decay, 'Weight Initialization': weight_init, 'Optimizer': opt.__name__, 'Model Type': model_type}
            best_performance['model'] = model

    epoch_end_time = time.time()
    elapsed_time = timedelta(seconds=epoch_end_time - epoch_start_time)
    log(APP_NAME, f"[SEQ({ii:03d})] Elapsed Time for Experiment: {elapsed_time}")


def run_full_experiment(num_epochs, batch_sizes, learning_rates, layer_depths, hidden_sizes, activation_funcs, dropout_rates, weight_decays, weight_inits, optimizers, models, criterion):
    best_performance = {'accuracy': 0, 'hyperparameters': None, 'model': None}

    total_start_time = time.time()  # 전체 러닝 타임 측정을 위한 시작 시간 기록

    log(APP_NAME, "Start to grid search..!")
    ii = 0

    # 모든 하이퍼파라미터 조합(product)에 대해 "그리드 서치" 실행
    for epoch, batch_size, lr, ld, hs, af, drr, weight_decay, weight_init, opt, model_type in product(num_epochs, batch_sizes, learning_rates, layer_depths, hidden_sizes, activation_funcs, dropout_rates, weight_decays, weight_inits, optimizers, models):
        ii += 1
        run_experiment(ii, epoch, batch_size, lr, ld, hs, af, drr, weight_decay, weight_init, opt, model_type, criterion, best_performance)

    total_end_time = time.time()  # 전체 러닝 타임 측정을 위한 종료 시간 기록
    total_elapsed_time = timedelta(seconds=total_end_time - total_start_time)  # 전체 러닝 타임 계산
    log(APP_NAME, f"Total Elapsed Time: {total_elapsed_time}")

    return best_performance


def Main():
    try:
        """
        런타임 설정
        """
        DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        log(APP_NAME, f'Using PyTorch version: {torch.__version__,} Device: {DEVICE}')

        """
        HyperParameter 설정
        """
        num_epochs = [5]
        batch_sizes = [32, 64, 128, 256]
        learning_rates = [0.1, 0.01, 0.001, 0.0001]
        layer_depths = [2, 3, 4, 5]
        hidden_sizes = [128, 256, 512, 1024]
        activation_funcs = ['relu', 'sigmoid', 'tanh', 'elu', 'leaky_relu', 'selu', 'softmax']
        dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
        weight_decays = [0.0, 1e-5, 1e-4, 1e-3]
        criterion = nn.CrossEntropyLoss()
        optimizers = [optim.Adam, optim.SGD, optim.RMSprop]
        weight_inits = ['uniform', 'normal', 'xavier', 'he', 'kaiming']

        """
        MODEL 설정
        """
        models = ['cnn', 'mlp']


        """
        실험 시작
        """
        best_performance = run_full_experiment(num_epochs, batch_sizes, learning_rates, layer_depths, hidden_sizes, activation_funcs, dropout_rates, weight_decays, weight_inits, optimizers, models, criterion)

        """
        가장 좋은 성능을 달성한 하이퍼파라미터와 모델 정보 출력
        """
        log(APP_NAME, f"Best Performance: Accuracy: {best_performance['accuracy']:.2f}%, Hyperparameters: {best_performance['hyperparameters']}, Model: {best_performance['model']}")

    except Exception as err:
        traceback_error = traceback.format_exc()
        log(APP_NAME, traceback_error)
        sys.exit()

if __name__ == "__main__":
    Main()
