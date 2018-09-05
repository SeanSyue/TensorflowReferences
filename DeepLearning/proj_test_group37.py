"""
@ COPYRIGHT: 蘇政柏、薛宇辰

# 這是測試檔案
# 請先修改參數ROOT為DeepLearningProject_group37資料夾路徑
# 除了需要修改參數ROOT，其餘的參數無需修改。
# 輸出結果顯示在.~/analysis/test_output.csv檔案中。
# 此程式運行結束之後，打開test_output.csv之後需把結果貼至評估表檔案analyze_template.xlsx中的相應欄位
# 每次運行完，都會覆蓋原先存在的test_output.csv檔案。

:param ROOT： 路徑名稱
:param TEST_FILE: 測試資料集位置
:param EXPORT_FILE: 儲存測試結果的檔案位置
:param CHECKPOINT_PATH： 訓練好的模型的存檔位置
:param FEATURE_COUNT： 資料集中「特徵」的數目
:param LABEL_COUNT： 資料集中「標籤」的數目
:param NODE_LIST：各層神經元的個數，從左到右依次代表輸入、各隱含層和輸出的神經元個數

"""
import tensorflow as tf
# -----------------------------------------------↓↓↓參數與預設函數↓↓↓-----------------------------------------------
from DeepLearning.ProjectFinal.proj_train_group37 \
    import neural_network, data_splitter, FEATURE_COUNT, LABEL_COUNT, NODE_LIST
# 直接從proj_train_group37.py導入函數參數，需要將proj_train_group37.py檔案放在可以被識別的python專案路徑之下
# 若無法導入，可刪除或comment起來上面一行程式碼，
# 再把proj_train_group37.py中預設函數neural_network, data_splitter，和變數NODE_LIST, FEATURE_COUNT, LABEL_COUNT複製貼上在這裡就好

ROOT = 'C:/DeepLearningProject_group37/'  # 運行前確認這個路徑是否正確
TEST_FILE = ROOT+'/dataset/data_test.csv'
EXPORT_FILE = ROOT+'analyze/test_output.csv'
CHECKPOINT_FILE = ROOT+'check_point/model_bank_500000.ckpt'  # 選取最後一次的存檔結果


# -----------------------------------------------↑↑↑參數與預設函數↑↑↑-----------------------------------------------


# -----------------------------------------------↓↓↓主程式↓↓↓-----------------------------------------------

def main():
    # 創建與訓練時一樣的Placehold和神經網路架構
    reshape_features, _, key = data_splitter(TEST_FILE, FEATURE_COUNT, LABEL_COUNT)
    x = tf.placeholder(tf.float32, [None, FEATURE_COUNT])
    target_conv = neural_network(x, NODE_LIST)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # 讀取訓練好的模型
        saver.restore(sess, CHECKPOINT_FILE)

        #  測試資料共8000筆，並輸出測試結果
        print("Analyzing...")
        output_ = open(EXPORT_FILE, mode='w+', encoding='utf-8')
        for _ in range(8000):
            predict = target_conv.eval(feed_dict={x: reshape_features.eval()})
            print("%g" % predict, file=output_, flush=True)
        output_.close()

        coord.request_stop()
        coord.join(threads)
    print("Finish!")


if __name__ == '__main__':
    main()
