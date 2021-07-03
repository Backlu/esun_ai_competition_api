## API設計心法
- API架構
    - 最底層是時下最時尚的API模組FasiAPI, 站在巨人的肩膀上一直是我們團隊的成功心法. FastAPI的內部架構就不贅述了, 人生苦短, 我用Python.
    - 中層是辨識影像內中文字的核心功能, 這部分功能請參考Source code內的run_inference.py
    - 最上程式API的出入口門戶, 接受辨識需求的輸入與答案的回傳. 這部分功能請參考Source code內的app.py

<P Align=center><img src="https://github.com/Backlu/esun_ai_competition_api/blob/main/api_framework.png" width="40%" alt="API架構"></p>

- API運作流程
    - End to End的運作流程也是我們在設計API時的心血結晶, 一定要在這裡分享給大家. 流程中的每一個方塊都是團隊裡每一位Data Enginnering, Data Scienties的心血結晶, 我們也不吝嗇的將每一部分的Source Code都分享在這個github repo裡了. 想要我的寶藏嗎? 如果想要的話, 那就到Source code去找吧, 我全部都放在那裡. 

<P Align=center><img src="https://github.com/Backlu/esun_ai_competition_api/blob/main/end2end.png" width="90%" alt="End2End"></p>


## 雲端平台
- 多快好省
    - 又好又穩: 在6/15測試賽的API斷線災情中, 完全不受影響! 
    - 又多又快: 最多可配置4張V100 GPU卡, 想多快就有多快!
    - 省錢省力: 每小時86就能使用原價30多萬的V100, 有錢還買不到的V100!
<P Align=center><img src="https://github.com/Backlu/esun_ai_competition_api/blob/main/api_borken.png" width="50%" alt="API斷線災情"></p>    

- 簡單好用
  - 自帶jupyter notebook & ssh功能. 超貼心! 
- 系統資源監控
  - 圖形化介面監控CPU與GPU的運算資源走勢, 在正式賽期間分分秒掌控運算資源變化, 比看股票還刺激啊! 
- 無痛備份
  - TWCC使用docker image的方式, 讓使用者one click無痛備份每一個程式與模型版本 
<P Align=center><img src="https://github.com/Backlu/esun_ai_competition_api/blob/main/twcc_good.png" width="60%" alt="TWCC"></p>


## Test Log
- 凡走過必留下痕跡. 每每在slack討論看到有人跪求測試資料就令我覺得痛心. Data絕對不是簡單打個show me the data就能無中生有的, 所以在這裡特別將保存測試資料的Source Code分享給大家. 
    - source code: 請參閱app.py內的save_input_image()與fastapi_logger

<P Align=center><img src="https://github.com/Backlu/esun_ai_competition_api/blob/main/show_me_the_data.png" width="50%" alt="Log"></p>

## 徵才啟事
- 看到最後, 相信你也很了解我們團隊的專案開發風格. 我們是一個持續有機成長中的AI大數據團隊, 專注在工業4.0領域的影像/NLP/大數據分析技術開發. 誠摯邀請各路英雄好漢加入! 有意請飛鴿傳書至: jayiios1105@gmil.com.  

