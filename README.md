# 📈 CPI Nowcasting with Mixed-Frequency Data and PCA-CSR Approach

本專案旨在建立一個精準的消費者物價指數 (CPI) 即時預測模型 (Nowcasting)。為了解決不同頻率數據的問題，本專案將「高頻率月資料」對齊並預測「低頻率季資料」。模型結合了 **Exponential Almon Weights** 權重函數、**主成分分析 (PCA)** 以及 **完全子集迴歸 (CSR, Complete Subset Regressions)** 的模型平均 (Model Averaging) 技巧。

## 📂 資料來源
本專案的數據皆來自美國聯邦準備經濟資料庫 (FRED)：
* **應變數 (Y)：** 低頻率季資料，FRED 的 CPI 季資料 (1959 Q1 至 2024 Q3)。在模型中計算為年增率 (YoY, 變動期數為 4)。
* **自變數 (X)：** 高頻率月資料。初步納入包含但不限於以下指標：
  * `AWHMAN`：製造業工人每週的平均工時
  * `TB3SMFFM`, `TB6SMFFM`, `T5YFFM`, `T10YFFM`：各天期國庫券收益率減去聯邦基金利率 (利差)
  * `AAAFFM`：穆迪評級為 AAA 的公司債收益率減去聯邦基金利率

## 🛠️ 模型架構與預測步驟

預測模型公式為： $y_t = \alpha_{PCA} + \beta_{PCA} \sum PCA_{1, t,k} \cdot w_k$

具體執行步驟如下：
1. **特徵篩選**：計算高頻月資料與 CPI 的相關係數，僅保留絕對值高於 0.4 的變數。
2. **完全子集分組 (CSR)**：將篩選出的變數，每次抓取 3 個變數進行組合，得到 $C^N_3$ 個組合。
3. **主成分分析 (PCA)**：在每個 3 變數組合中進行 SVD 奇異值分解，萃取最具解釋力的第一主成分 (PCA1) 作為新的自變數。
4. **非線性最佳化 (NLS)**：定義 Exponential Almon Weights 作為權重函數 $w_k$，並使用 `L-BFGS-B` 演算法優化 $\theta_1, \theta_2, \beta, \alpha$ 四個參數，目標為最小化殘差平方和 (RSS)。每個組合進行 2 次隨機初始化的訓練，取最佳結果。
5. **模型平均 (Ensemble/Averaging)**：將所有組合預估出的 $\hat{y}_t$ 進行簡單平均，得出最終的預測結果，以降低單一模型的變異與過擬合風險。

## 💻 技術棧 (Tech Stack)
* **程式語言:** Python
* **資料處理:** Pandas, NumPy
* **數值運算與最佳化:** SciPy (`scipy.optimize.minimize`, `numpy.linalg.svd`)
* **評估與視覺化:** Scikit-learn (`mean_squared_error`, `mean_absolute_error`, `r2_score`), Matplotlib

## 🚀 如何執行專案

1. **安裝依賴套件:**
   ```bash
   pip install numpy pandas matplotlib scipy scikit-learn