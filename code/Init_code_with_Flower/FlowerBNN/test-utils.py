from Net import *
from utils import *
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Binary_CF_Mini().to(DEVICE)
if __name__ == "__main__":
    mdpth = r'D:\Graduation project\Code\FlowerBNN\results\models\\'[:-1]
    Modelname = 'BCFMini24.pth'
    net_dict = torch.load(mdpth+Modelname)
    net.load_state_dict(net_dict)

    flattened_weight,ori_shape = get_flat_weights(net)
    print('len:',len(flattened_weight))
    print("original_shape",ori_shape)
    padded_params, l = pad_to_power_of_2(flattened_weight)
    print("Initial l:",l)
    print("After Padding",len(padded_params),padded_params[:10])

    unflatten_w = unflatten_weights(flattened_weight,ori_shape)
    set_weight(net,unflatten_w)
    print([para for para in net.parameters()])
#不太对，我们应该只加密和传输需要训练的参数，使用dict这样把不用训练的参数也给弄进来了
#去改get_weights和set_weights不能用dict的方法