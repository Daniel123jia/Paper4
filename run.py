# 可以跑通 
python DN4_Train_5way1shot.py --dataset_dir ./dataset/MSTAR/ --data_name MSTAR
python DN4_Test_5way1shot.py --resume ./results/DN4_MSTAR_Conv64F_5Way_1Shot_K3/model_best.pth.tar --outf XXXX


python DN4_Test_5way1shot.py --resume ./results/DN4_MSTAR_Conv64F_5Way_1Shot_K3/model_best.pth.tar # 测试可以跑通 
还有两个（ReseNet）的没有训练和测试，由于GPU显存不够用，故暂停……
