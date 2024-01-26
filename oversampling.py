


    for i in range(0,8):
        for j in range(0,2):
            PATH = "./compdata/bdhsc_2024/stage1_labeled/" + str(i) + "_" + str(j) + ".csv"
            data = pd.read_csv(PATH)