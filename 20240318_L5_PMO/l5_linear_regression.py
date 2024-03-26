import sys
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class LinearRegression(nn.Module):
    def __init__(self, d_input, d_output):
        # 부모 클래스(nn.Module) 상속
        super(LinearRegression, self).__init__()
        self.d_input = d_input
        self.d_output = d_output

        """
        [nn.Parameter]
         학습변수를 관리하기 위한 컨테이너이다. nn.Parameter로 세팅된 변수는 'requires_grad'가 기본으로 True이다.
        """
        self.W_model = nn.Parameter(torch.randn(d_input, d_output))

    def forward(self, X):
        return X @ self.W_model


def get_data(d_input, d_output, n, noise, learning_rate, n_samples):
    try:
        X = torch.randn(n, d_input)
        W_true = torch.randn(d_input, d_output)
        Y = X @ W_true
        Y = Y + torch.randn(n, d_output) * noise

        return X, W_true, Y
    except Exception as err:
        raise (err)


def plot(target, title, xlabel, ylabel, figsize=(5, 5)):
    plt.figure(figsize=figsize)
    plt.plot(target)
    plt.title(title)  # Corrected assignment
    plt.xlabel(xlabel)  # Corrected assignment
    plt.ylabel(ylabel)  # Corrected assignment
    plt.xlim(0, 1000)
    plt.ylim(0, max(target))  # Specify upper limit for y-axis
    plt.show()


def main():
    try:
        # Setting Parameters
        d_input = 100
        d_output = 10
        n = d_input * d_output  # d_input * d_output
        noise = 0.01
        learning_rate = 0.1  # 학습률
        n_samples = 10  # 샘플 개수

        # X, W_true, Y 가져오기
        X, W_true, Y = get_data(d_input, d_output, n,
                                noise, learning_rate, n_samples)

        # 모델 클래스 선언
        model = LinearRegression(d_input, d_output) # 직접 선언

        """
        [nn.Linear]
         Pytorch에서 제공하는 신경망 모듈 중 하나로, 선형변환을 수행한다.
         
         <필드 설명>
          - in_features: 입력 특징의 수입니다. 이 값은 입력 데이터의 차원이며, 입력 데이터의 각 특징에 해당하는 차원입니다.
          - out_features: 출력 특징의 수입니다. 이 값은 출력 데이터의 차원이며, 계층이 생성하는 출력의 차원을 결정합니다.
          - bias: 옵션으로, 편향(bias)을 사용할지 여부를 나타내는 불리언 값입니다. 기본값은 True이며, 이 경우에는 선형 변환의 결과에 편향이 추가됩니다.
        """
        # model = nn.Linear(d_input, d_output, bias=False)  # 이미 만들어진 클래스 메소드 사용

        # 최적화 알고리즘
        """
        [optim.SGD]
         확률적 경사 하강법은 가장 기본적인 최적화 알고리즘 중 하나입니다. 
         이 알고리즘은 각 가중치 매개변수에 대해 손실 함수의 기울기(경사)를 계산하고, 기울기의 반대 방향으로 가중치를 업데이트합니다.
         
         <필드 설명>
          - params: 최적화할 매개변수의 Iterable 객체입니다.
          - lr: 학습률(learning rate)로, 각 가중치를 업데이트할 때 사용되는 스케일링된 학습 속도를 나타냅니다.
          - momentum (옵션): 모멘텀(momentum) 매개변수는 SGD의 변형으로, 이전 그래디언트 업데이트를 고려하여 다음 업데이트에 추가합니다. 이를 통해 SGD의 수렴 속도를 높일 수 있습니다.
        """
        # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9) # SGD(경사 하강법)

        """
        [optim.Adam]
         Adam은 RMSProp과 모멘텀의 아이디어를 결합한 최적화 알고리즘으로, 더욱 정교한 학습률 조절을 가능하게 합니다. 
         Adam은 각 매개변수에 대해 개별적인 학습률을 제공하며, 학습률을 적응적으로 조정하여 각 매개변수에 대한 경사의 이동 평균 및 이동 제곱 평균을 추정합니다.

         <필드 설명>
          - params: 최적화할 매개변수의 Iterable 객체입니다.
          - lr: 학습률(learning rate)로, 각 가중치를 업데이트할 때 사용되는 스케일링된 학습 속도를 나타냅니다.
          - betas (옵션): Adam 알고리즘에서 사용되는 이동 평균을 계산하는 데 사용되는 두 개의 계수를 정의합니다. 기본값은 (0.9, 0.999)입니다.
        """
        optimizer = optim.Adam(model.parameters(), lr=1e-1)  # ADAM

        # 손실함수
        loss_func = nn.MSELoss()

        # 손실율 체크 리스트
        loss_hist = []

        # 손실 및 가중치 변화 추적 리스트 초기화
        W_diff_hist = []

        """
        [반복 샘플링]
         이 접근 방식에서는 고정된 횟수의 반복(이 경우 1000번)을 사용하며, 
         각 반복마다 0부터 n-1까지의 인덱스를 무작위로 섞은 후에 처음부터 n_samples개의 인덱스를 선택합니다. 
         이로써 각 반복에서 전체 데이터셋에서 무작위로 미니 배치를 샘플링합니다. 그러나 동일한 에포크 내에서 동일한 데이터 포인트가 여러 번 샘플링될 수 있습니다.
        """
        # for ii in range(1000):
        #     # 샘플링
        #     idx = torch.randperm(n)
        #     idx_samples = idx[:n_samples]
        #     Y_sampled = Y[idx_samples, :]
        #     X_sampled = X[idx_samples, :]
        #     Y_predict = model(X_sampled)
        #     loss = loss_func(Y_sampled, Y_predict)
        #     loss.backward()

        #     """
        #     - torch.no_grad()를 하는 이유:
        #     W_model의 기울기를 추적하도록 세팅하면, 원본 값을 직접 수정할 때 에러가 발생한다.
        #     그래서 기울기를 잠시 추적하지 않겠다는 torch.no_grad()가 필요하다.
        #     """
        #     # with torch.no_grad():
        #     #     W_model -= learning_rate * W_model.grad # 경사 하강법(W_model 이 점점 이동)

        #     optimizer.step()
        #     optimizer.zero_grad()

        #     # 리스트에 추가
        #     loss_hist.append(loss.item())
        #     W_diff_hist.append(torch.norm(W_true - model.W_model).item())

        """
        [Epoch 기반 샘플링]
         이 접근 방식에서는 각 에포크마다 0부터 n-1까지의 인덱스를 무작위로 섞은 후에 해당 인덱스를 사용하여 데이터를 미니 배치로 나눕니다. 
         이렇게 함으로써 각 에포크에서 전체 데이터셋을 무작위로 볼 수 있으며, 동일한 에포크 내에서 중복 샘플링이 발생하지 않습니다.
        """
        for epoch in range(10):
            idx_random = torch.randperm(n)

            for ii in range(n // n_samples):
                idx_samples = idx_random[ii * n_samples: (ii + 1) * n_samples]

                Y_sampled = Y[idx_samples]
                X_sampled = X[idx_samples]

                Y_predicted = model(X_sampled)

                loss = loss_func(Y_sampled, Y_predicted)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                loss_hist.append(loss.item())
                W_diff_hist.append(torch.norm(W_true - model.W_model).item())

        # Plot 생성
        plot(loss_hist, 'Loss', 'Iteration', 'Loss')
        plot(W_diff_hist, 'Difference between W_true and W_model',
             'Iteration', 'Difference')
    except Exception as err:
        print(err)
        sys.exit()


if __name__ == "__main__":
    main()
