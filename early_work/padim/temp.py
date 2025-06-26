from mvtec_loader import MVTecDataset
from matplotlib import pyplot as plt

dataset = MVTecDataset(root_path='data/mvtec_anomaly_detection', category='bottle', is_train=True)

print(len(dataset))

for i in range(2):
    img, mask, label = dataset[i]
    plt.figure()
    img = img.numpy()
    img = img.transpose(1, 2, 0)
    plt.imshow(img)
    plt.title('Original')
    plt.axis('off')
    plt.show()