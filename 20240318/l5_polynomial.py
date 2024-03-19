import sys
import torch
import matplotlib.pyplot as plt

def get_data(degree, n, noise):
    try:
        x = torch.randn(n) / 2 # (n x 1)
        a_true = torch.randn(degree + 1) # a3 a2 a1 a0
        y = torch.zeros(n)
        a_model = torch.randn(degree + 1, requires_grad=True)

        """
        1) y = a0x^0
        2) y = a0x^0 + a1x^1
        3) ...
        """
        for ii in range(degree + 1):
            y = y + x ** ii * a_true[ii]

        return x, a_true, y, a_model
    except Exception as err:
        raise(err)

def model(a_model, degree, x):
    y = torch.zeros(x.shape)
    for ii in range(degree + 1):
        y = y + x ** ii * a_model[ii]

    return y

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
        # Setting parameters
        degree = 3
        n = 100
        noise = 0.01
        learning_rate = 5e-2
        n_batch = 10 # number of samples in each bach

        x, a_true, y, a_model = get_data(degree, n, noise)

        loss_hist = []
        a_diff_hist = []

        for ii in range(1000):
            idx = torch.randperm(n)
            idx_samples = idx[:n_batch]

            y_sampled = y[idx_samples]
            x_sampled = x[idx_samples]

            ## 선형 회귀 모델
            y_predicted = model(a_model, degree, x_sampled)
           
            ## 손실 함수
            loss = torch.mean((y_sampled - y_predicted) ** 2)
            loss.backward()

            ## 최적화 과정
            with torch.no_grad():
                a_model -= learning_rate * a_model.grad    
            a_model.grad.zero_()

            loss_hist.append(loss.item())
            a_diff_hist.append(torch.norm(a_model - a_true).item())
        
        plot(loss_hist, 'Loss', 'Iteration', 'Loss')
        plot(a_diff_hist, 'Difference between W_true and W_model',
             'Iteration', 'Difference')
    except Exception as err:
        print(err)
        sys.exit()

if __name__ == "__main__":
    main()