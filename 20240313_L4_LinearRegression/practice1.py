import torch

def get_data(d_input, d_output, n, noise):
    X = torch.randn(n, d_input)  # n x d_input
    W_true = torch.randn(d_input, d_output)  # d_input x d_output
    Y = X @ W_true  # ground truth(n x d_output)
    Y = Y + torch.randn(n, d_output) * noise  # adding noise
    
    W_model = torch.randn(d_input, d_output, requires_grad=True)
    return X, Y, W_true, W_model

def main():
    X, Y, W_true, W_model = get_data(100, 10, 1000, 0.01)

    learning_rate = 0.01
    num_epochs = 1000

    for epoch in range(num_epochs):
        index = torch.randperm(X.size(0)) # X 행 개수(n) 만큼 랜덤 값 리턴
        index_sampled = index[:10] # 그 중에서 10개만 가져옴
        X_sampled = X[index_sampled, :] # 랜덤한 10개의 행으로 샘플 값 X 도출 => 10 x d_input
        Y_sampled = Y[index_sampled, :] # 랜덤한 10개의 행으로 샘플 값 Y 도출 => 10 x d_output

        Y_pred = X_sampled @ W_model # 샘플 값 X(10 x d_input)와 W_model(d_input x d_output) 행렬 연산 => Y_pred = 10 x d_output

        # 손실 함수
        loss = torch.mean((Y_sampled - Y_pred)**2) # (1/|B|)sigma(||y- Wx||^2) => ||y- Wx||^2 의 평균을 나타냄

        loss.backward() # 손실함수에 대한 기울기 계산

        """
        - torch.no_grad()를 하는 이유:
          W_model의 기울기를 추적하도록 세팅하면, 원본 값을 직접 수정할 때 에러가 발생한다. 
         그래서 기울기를 잠시 추적하지 않겠다는 torch.no_grad()가 필요하다.
        """
        with torch.no_grad():
            W_model -= learning_rate * W_model.grad # 경사 하강법(W_model 이 점점 이동)
        
        """
        Autograd는 기울기를 누적하여 저장한다. 초기화 하지 않으면 이전 반복에서의 기울기 값이 현재 반복에서 계산된 기울기 값에 누적된다.
        """
        W_model.grad.zero_() # 다음 반복을 위해 기울기 초기화

        # 진행상황 출력
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # W의 추정값 확인
    print('True W:')
    print(W_true)
    print('\nEstimated W:')
    print(W_model)

if __name__ == "__main__":
    main()
