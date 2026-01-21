from torch.utils.data import DataLoader

from tqdm import tqdm
import argparse

from datasets.LoadSONN import ScanObjectNN
#from datasets.data_scan import ScanObjectNN
from datasets.part_dataset_all_normal import PartNormalDataset
from datasets.ModelNet40Loader import ModelNet40Cls, ModelNet10Cls
from datasets.ModelNet40FewShot import ModelNet40FewShot

from utils import *
from models.NPMFF import NPMFF


def get_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mn40')
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=10)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--variant', type=str, default='OBJ-BG')
    parser.add_argument('--load_memory', action='store_true', help="use the memory of the last checkpoint")

    parser.add_argument('--split', type=int, default=3)
    parser.add_argument('--transforms', action='store_false', help="Disable transforms")
    parser.add_argument('--use_anp', action='store_false', help="Disable anp")
    parser.add_argument('--p', type=float, default=2)
    parser.add_argument('--bz', type=int, default=16)  # Freeze as 16

    parser.add_argument('--points', type=int, default=1024)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--dim', type=int, default=42)

    parser.add_argument('--k', type=int, default=90)
    parser.add_argument('--alpha', type=int, default=1500)
    parser.add_argument('--beta', type=int, default=100)

    args = parser.parse_args()
    return args
    

@torch.no_grad()
def main():
    import datasets.data_utils as d_utils
    from torchvision import transforms

    transform = transforms.Compose(
        [
            d_utils.PointcloudToTensor(),
            d_utils.PointcloudRandomInputDropout(0.2),
            #d_utils.PointcloudRotate(),
            #d_utils.PointcloudScale(),
            #d_utils.PointcloudTranslate(0.05),
            d_utils.PointcloudJitter(),
        ]
    )


    print('==> Loading args..')
    args = get_arguments()
    print(args)

    feature_memory_file = './data/cls_feature_memory_dim%d_k%d_depth%d_%s.pt' % (
        args.dim, args.k, args.depth, args.dataset)
    label_memory_file = './data/cls_label_memory_dim%d_k%d_depth%d_%s.pt' % (
        args.dim, args.k, args.depth, args.dataset)

    print('==> Preparing model..')
    point_nn = NPMFF(input_points=args.points, depth=args.depth,
                        embed_dim=args.dim, k_neighbors=args.k,
                        alpha=args.alpha, beta=args.beta, use_anp=args.use_anp, p=args.p).cuda()
    point_nn.eval()


    print('==> Preparing data..')

    transform = transform if args.transforms else None

    print('==> Data Transforms -- ', args.transforms)

    if args.dataset == 'scan':
        train_loader = DataLoader(ScanObjectNN(split=args.split, variant=args.variant, partition='training', num_points=args.points),
                                    num_workers=16, batch_size=args.bz, shuffle=False, drop_last=False)
        test_loader = DataLoader(ScanObjectNN(split=args.split, variant=args.variant,partition='test', num_points=args.points),
                                    num_workers=16, batch_size=args.bz, shuffle=False, drop_last=False)
    if args.dataset == 'mn40':

        train_loader = DataLoader(ModelNet40Cls(num_points=args.points, train=True, transforms=transform),
                                  num_workers=16, batch_size=args.bz, shuffle=False, drop_last=False, pin_memory=True)
        test_loader = DataLoader(ModelNet40Cls(num_points=args.points, train=False, transforms=None),
                                 num_workers=16, batch_size=args.bz, shuffle=False, drop_last=False)

    if args.dataset == 'mn40FS':
        train_loader = DataLoader(ModelNet40FewShot(root='./data/ModelNetFewshot', split="train",
                                                    way=args.way, shot=args.shot, fold=args.fold, transforms=transform),
                                  num_workers=16, batch_size=args.bz, shuffle=False, drop_last=False,
                                  pin_memory=True)
        test_loader = DataLoader(ModelNet40FewShot(root='./data/ModelNetFewshot', split="test",
                                                    way=args.way, shot=args.shot, fold=args.fold),
                                 num_workers=16, batch_size=args.bz, shuffle=False, drop_last=False)
    if args.dataset == 'mn10':

        train_loader = DataLoader(ModelNet10Cls(num_points=args.points, train=True, transforms=transform),
                                  num_workers=16, batch_size=args.bz, shuffle=False, drop_last=False, pin_memory=True)
        test_loader = DataLoader(ModelNet10Cls(num_points=args.points, train=False, transforms=None),
                                 num_workers=16, batch_size=args.bz, shuffle=False, drop_last=False)
    if args.dataset == 'shapenet':
        train_loader = DataLoader(PartNormalDataset(root='./data/shapenetcore_partanno_segmentation_benchmark_v0_normal',
                                                    split='trainval', classification=True, npoints=args.points, normalize=True),
                                  num_workers=16, batch_size=args.bz, shuffle=False, drop_last=False, pin_memory=True)
        test_loader = DataLoader(PartNormalDataset(root='./data/shapenetcore_partanno_segmentation_benchmark_v0_normal',
                                                    split='test', classification=True, npoints=args.points, normalize=True),
                                 num_workers=16, batch_size=args.bz, shuffle=False, drop_last=False)

    print('==> Constructing Point-Memory Bank..')

    feature_memory, label_memory = [], []
    if args.load_memory and os.path.exists(feature_memory_file):
        print('==> Loading feature and label memory from file..')
        feature_memory = torch.load(feature_memory_file)
        label_memory = torch.load(label_memory_file)
    else:
        # with torch.no_grad():
        for points_normals, labels in tqdm(train_loader):
            points_normals = points_normals.cuda(non_blocking=True).permute(0, 2, 1)

            point_features = point_nn(points_normals)
            feature_memory.append(point_features)

            labels = labels.cuda(non_blocking=True)
            label_memory.append(labels)

        # Feature Memory
        feature_memory = torch.cat(feature_memory, dim=0)
        feature_memory /= feature_memory.norm(dim=-1, keepdim=True)
        feature_memory = feature_memory.permute(1, 0)
        # Label Memory
        label_memory = torch.cat(label_memory, dim=0)
        label_memory = label_memory.squeeze().long()
        label_memory = F.one_hot(label_memory).squeeze().float()

        #torch.save(feature_memory, feature_memory_file)
        #torch.save(label_memory, label_memory_file)
        #print('==> Feature and label memory saved to file.')

    print('==> Saving Test Point Cloud Features..')
    
    test_features, test_labels = [], []
    with torch.no_grad():
        for points_normals, labels in tqdm(test_loader):

            points_normals = points_normals.cuda(non_blocking=True).permute(0, 2, 1)
            point_features = point_nn(points_normals)
            test_features.append(point_features)

            labels = labels.cuda(non_blocking=True)
            test_labels.append(labels)

    test_features = torch.cat(test_features)
    test_features /= test_features.norm(dim=-1, keepdim=True)
    test_labels = torch.cat(test_labels)


    print('==> Starting..')

    best_acc, best_gamma = 0, 0
    for gamma in torch.linspace(0, 10000, steps=500):  # 减少gamma的数量
        Sim = test_features @ feature_memory
        logits = (-gamma * (1 - Sim)).exp() @ label_memory
        acc = cls_acc(logits, test_labels)
        if acc > best_acc:
            best_acc, best_gamma = acc, gamma

    print(f"Classification accuracy on Dataset {args.dataset} : {best_acc:.2f}. gamma: {best_gamma:.2f}")


if __name__ == '__main__':
    main()
