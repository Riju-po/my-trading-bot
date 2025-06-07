# trading_bot.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import os
from datetime import datetime, timedelta
import ta
from sklearn.preprocessing import StandardScaler
import asyncio
import requests # For KrakenDataFetcher
import time     # For KrakenDataFetcher rate limit sleep
import krakenex # For KrakenDataFetcher

# --- Telegram Bot Imports ---
try:
    from telegram import Bot
    from telegram.constants import ParseMode
    TELEGRAM_AVAILABLE = True
except ImportError:
    print("WARNING: python-telegram-bot library not found. Telegram alerts will be simulated.")
    TELEGRAM_AVAILABLE = False
    class Bot: pass # Dummy class
    class ParseMode: HTML = "HTML"; MARKDOWN_V2="MarkdownV2" # Dummy class

print("--- Initializing Trading Bot Script ---")

# --- CLASS DEFINITIONS ---

# 1. KrakenDataFetcher (from your Cell 2)
class KrakenDataFetcher:
    def __init__(self):
        try:
            self.api = krakenex.API()
        except Exception as e:
            print(f"Error initializing krakenex.API(): {e}")
            raise
        self.base_url = "https://api.kraken.com/0/public/"
        self.pair = "XXBTZUSD"

    def get_ohlc_data(self, interval_minutes=15, since_timestamp=None, retries=3, backoff_factor=2):
        kraken_interval_map = {1:1, 5:5, 15:15, 30:30, 60:60, 240:240, 1440:1440, 10080:10080, 21600:21600}
        kraken_interval = kraken_interval_map.get(interval_minutes)
        if kraken_interval is None:
            print(f"Warning: Interval {interval_minutes}m not supported, using 15m."); kraken_interval = 15
        params = {'pair': self.pair, 'interval': kraken_interval}
        if since_timestamp: params['since'] = int(since_timestamp)
        for attempt in range(retries):
            try:
                url = f"{self.base_url}OHLC"
                response = requests.get(url, params=params, timeout=20)
                response.raise_for_status()
                data = response.json()
                if data.get('error') and len(data['error']) > 0:
                    print(f"Kraken API Error (Attempt {attempt+1}): {data['error']}")
                    if "EAPI:Rate limit exceeded" in str(data['error']):
                        time.sleep(backoff_factor * (attempt+1)); continue
                    return pd.DataFrame()
                actual_pair_key = list(data['result'].keys())[0] if data.get('result') else None
                if not actual_pair_key: return pd.DataFrame()
                ohlc_list = data['result'][actual_pair_key]
                if not isinstance(ohlc_list, list): return pd.DataFrame()
                df = pd.DataFrame(ohlc_list, columns=['timestamp','open','high','low','close','vwap','volume','count'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
                num_cols = ['open','high','low','close','vwap','volume']
                for col in num_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
                df['count'] = pd.to_numeric(df['count'], errors='coerce').fillna(0).astype(int)
                df.dropna(subset=num_cols, inplace=True)
                if df.empty: return pd.DataFrame()
                df.set_index('timestamp', inplace=True); df.sort_index(inplace=True)
                return df
            except requests.exceptions.RequestException as e: print(f"HTTP Error (Attempt {attempt+1}): {e}")
            except Exception as e: print(f"Unexpected error in get_ohlc_data (Attempt {attempt+1}): {e}")
            if attempt == retries - 1: return pd.DataFrame()
            time.sleep(backoff_factor * (attempt+1))
        return pd.DataFrame()

# 2. SmartMoneyAnalyzer (from your Cell 4)
class SmartMoneyAnalyzer:
    def __init__(self, df_ohlc):
        if not isinstance(df_ohlc, pd.DataFrame) or df_ohlc.empty: raise ValueError("Input df_ohlc invalid.")
        if not all(c in df_ohlc.columns for c in ['open','high','low','close','volume']): raise ValueError("Missing OHLCV columns.")
        self.df = df_ohlc.copy(); self.lookback_period = 20; self.volume_threshold_multiplier = 1.5
        if len(self.df) >= self.lookback_period:
            self.df['vol_avg']=self.df['volume'].rolling(window=self.lookback_period,min_periods=10).mean()
            self.df['high_volume_flag']=self.df['volume']>(self.df['vol_avg']*self.volume_threshold_multiplier)
        else: self.df['vol_avg']=np.nan; self.df['high_volume_flag']=False
        if len(self.df) >= 3:
            self.df['swing_high_flag']=(self.df['high']>self.df['high'].shift(1))&(self.df['high']>self.df['high'].shift(-1))
            self.df['swing_low_flag']=(self.df['low']<self.df['low'].shift(1))&(self.df['low']<self.df['low'].shift(-1))
        else: self.df['swing_high_flag']=False; self.df['swing_low_flag']=False
        for flag_col in ['high_volume_flag','swing_high_flag','swing_low_flag']:
            if flag_col in self.df.columns: self.df[flag_col]=self.df[flag_col].fillna(False)
    def detect_order_blocks(self): # Simplified from your Cell 4 logic
        # print("  SMC Analyzer: Detecting Order Blocks...") # Optional print
        bullish_obs, bearish_obs = [], []
        if not all(c in self.df.columns for c in ['vol_avg','high_volume_flag','swing_high_flag','swing_low_flag']): return {'bullish':[],'bearish':[]}
        for i in range(1, len(self.df)-1):
            row=self.df.iloc[i]; ts=self.df.index[i]
            if row['swing_high_flag'] and row['high_volume_flag'] and row['close']<row['open']: bearish_obs.append({'timestamp':ts,'type':'bearish_ob','high':row['high'],'low':row['low'],'volume':row['volume'],'strength':min(row['volume']/(row['vol_avg']+1e-9),5.0) if pd.notna(row['vol_avg']) else 1.0})
            if row['swing_low_flag'] and row['high_volume_flag'] and row['close']>row['open']: bullish_obs.append({'timestamp':ts,'type':'bullish_ob','high':row['high'],'low':row['low'],'volume':row['volume'],'strength':min(row['volume']/(row['vol_avg']+1e-9),5.0) if pd.notna(row['vol_avg']) else 1.0})
        return {'bullish':bullish_obs,'bearish':bearish_obs}
    def detect_liquidity_pools(self, lookback_lp=10, prox_pct=0.001):
        # print("  SMC Analyzer: Detecting Liquidity Pools...") # Optional print
        lps=[];
        if len(self.df)<lookback_lp:return lps
        for i in range(lookback_lp,len(self.df)):
            ch=self.df['high'].iloc[i];cl=self.df['low'].iloc[i];ts=self.df.index[i]
            rhs=self.df['high'].iloc[i-lookback_lp:i];rls=self.df['low'].iloc[i-lookback_lp:i]
            neh=rhs[abs(rhs-ch)<(ch*prox_pct)];nel=rls[abs(rls-cl)<(cl*prox_pct)]
            if len(neh)>=2:lps.append({'timestamp':ts,'type':'sell_side_liquidity_retest','price_level':ch,'strength':len(neh)})
            if len(nel)>=2:lps.append({'timestamp':ts,'type':'buy_side_liquidity_retest','price_level':cl,'strength':len(nel)})
        return lps
    def detect_fair_value_gaps(self,min_gap_pct=0.1):
        # print("  SMC Analyzer: Detecting Fair Value Gaps...") # Optional print
        fvgs=[];
        if len(self.df)<3:return fvgs
        for i in range(2,len(self.df)):
            h_im2=self.df['high'].iloc[i-2];l_im2=self.df['low'].iloc[i-2];h_i=self.df['high'].iloc[i];l_i=self.df['low'].iloc[i]
            ts=self.df.index[i];cp=self.df['close'].iloc[i]
            if l_i>h_im2:gs_p=l_i-h_im2;gp=(gs_p/(cp+1e-9))*100;
            if gp>=min_gap_pct:fvgs.append({'timestamp':ts,'type':'bullish_fvg','top':l_i,'bottom':h_im2,'gap_size_usd':gs_p,'gap_percentage':gp,'candle_idx':i})
            elif h_i<l_im2:gs_p=l_im2-h_i;gp=(gs_p/(cp+1e-9))*100;
            if gp>=min_gap_pct:fvgs.append({'timestamp':ts,'type':'bearish_fvg','top':l_im2,'bottom':h_i,'gap_size_usd':gs_p,'gap_percentage':gp,'candle_idx':i})
        return fvgs
    # Note: detect_market_structure_shifts from your Cell 4 was very simplified and might be too noisy.
    # If you want to include it, paste its definition here. For now, it's omitted for brevity.

# 3. SmartMoneyTradingSystem (from your Cell 6, with UnboundLocalError fix)
class SmartMoneyTradingSystem:
    def __init__(self, data): self.data=data.copy();self.signals=[];self.lookback_period=20
    def calculate_smc_indicators(self):
        df=self.data.copy()
        if len(df)<self.lookback_period+5:cols=['swing_high_flag','swing_low_flag','volume_sma','high_volume_flag','bullish_ob','bearish_ob','bullish_fvg','bearish_fvg','buy_side_liquidity','sell_side_liquidity','bullish_bos','bearish_bos'];[df.update({col:(0.0 if 'fvg' in col else 0)}) for col in cols if col not in df];return df.fillna(0)
        df['swing_high_flag']=(df['high']>df['high'].shift(1))&(df['high']>df['high'].shift(-1));df['swing_low_flag']=(df['low']<df['low'].shift(1))&(df['low']<df['low'].shift(-1));df['volume_sma']=df['volume'].rolling(self.lookback_period,min_periods=1).mean();df['high_volume_flag']=df['volume']>(df['volume_sma']*1.5)
        df['bullish_ob']=0;df['bearish_ob']=0
        for i in range(2,len(df)-2):
            if(df['close'].iloc[i]>df['open'].iloc[i] and df['high_volume_flag'].iloc[i] and df['swing_low_flag'].iloc[i]):df.loc[df.index[i],'bullish_ob']=1
            if(df['close'].iloc[i]<df['open'].iloc[i] and df['high_volume_flag'].iloc[i] and df['swing_high_flag'].iloc[i]):df.loc[df.index[i],'bearish_ob']=1
        df['bullish_fvg']=0.0;df['bearish_fvg']=0.0
        for i in range(2,len(df)):
            if df['low'].iloc[i]>df['high'].iloc[i-2]:gs=(df['low'].iloc[i]-df['high'].iloc[i-2])/(df['close'].iloc[i]+1e-9);df.loc[df.index[i],'bullish_fvg']=gs if gs>0.001 else(df.loc[df.index[i-1],'bullish_fvg']if i>0 and 'bullish_fvg'in df.columns else 0.0)
            if df['high'].iloc[i]<df['low'].iloc[i-2]:gs=(df['low'].iloc[i-2]-df['high'].iloc[i])/(df['close'].iloc[i]+1e-9);df.loc[df.index[i],'bearish_fvg']=gs if gs>0.001 else(df.loc[df.index[i-1],'bearish_fvg']if i>0 and'bearish_fvg'in df.columns else 0.0)
        df['buy_side_liquidity']=0;df['sell_side_liquidity']=0;w=10
        for i in range(w,len(df)):
            rl=df['low'].iloc[i-w:i];rh=df['high'].iloc[i-w:i];cl=df['low'].iloc[i];ch=df['high'].iloc[i];els=rl[abs(rl-cl)<(cl*0.002)];ehs=rh[abs(rh-ch)<(ch*0.002)]
            if len(els)>=2:df.loc[df.index[i],'buy_side_liquidity']=len(els)
            if len(ehs)>=2:df.loc[df.index[i],'sell_side_liquidity']=len(ehs)
        df['bullish_bos']=0;df['bearish_bos']=0;shp=df[df['swing_high_flag']]['high'];slp=df[df['swing_low_flag']]['low']
        for i in range(self.lookback_period,len(df)):
            ct=df.index[i];rhbc=shp[shp.index<ct].tail(2);rlbc=slp[slp.index<ct].tail(2)
            if len(rhbc)>=2 and df['high'].iloc[i]>rhbc.iloc[-1]:df.loc[ct,'bullish_bos']=1
            if len(rlbc)>=2 and df['low'].iloc[i]<rlbc.iloc[-1]:df.loc[ct,'bearish_bos']=1
        return df.fillna(0)
    def generate_buy_signals(self,df_smc):
        bs=[];
        if len(df_smc)<50:return bs
        for i in range(50,len(df_smc)):
            s=0;cr=[]
            if df_smc['bullish_ob'].iloc[i-5:i].sum()>0 and df_smc['low'].iloc[i]<=df_smc['low'].iloc[i-5:i].min()*1.002:s+=3;cr.append("Bullish_OB_Reaction")
            if df_smc['bullish_fvg'].iloc[i-3:i].sum()>0:s+=2;cr.append("Bullish_FVG")
            if df_smc['buy_side_liquidity'].iloc[i-2:i].sum()>0 and df_smc['close'].iloc[i]>df_smc['open'].iloc[i]:s+=2;cr.append("Liquidity_Sweep_Reversal")
            if df_smc['bullish_bos'].iloc[i]>0:s+=3;cr.append("Bullish_BOS")
            if df_smc['high_volume_flag'].iloc[i]:s+=1;cr.append("Volume_Confirmation")
            if df_smc['close'].iloc[i]>df_smc['open'].iloc[i]:s+=1;cr.append("Bullish_Candle")
            if s>=4:bs.append({'timestamp':df_smc.index[i],'type':'BUY','price':df_smc['close'].iloc[i],'score':s,'reasons':cr,'stop_loss':self.calculate_stop_loss(df_smc,i,'buy'),'take_profit':self.calculate_take_profit(df_smc,i,'buy')})
        return bs
    def generate_sell_signals(self,df_smc):
        ss=[];
        if len(df_smc)<50:return ss
        for i in range(50,len(df_smc)):
            s=0;cr=[]
            if df_smc['bearish_ob'].iloc[i-5:i].sum()>0 and df_smc['high'].iloc[i]>=df_smc['high'].iloc[i-5:i].max()*0.998:s+=3;cr.append("Bearish_OB_Reaction")
            if df_smc['bearish_fvg'].iloc[i-3:i].sum()>0:s+=2;cr.append("Bearish_FVG")
            if df_smc['sell_side_liquidity'].iloc[i-2:i].sum()>0 and df_smc['close'].iloc[i]<df_smc['open'].iloc[i]:s+=2;cr.append("Liquidity_Sweep_Reversal")
            if df_smc['bearish_bos'].iloc[i]>0:s+=3;cr.append("Bearish_BOS")
            if df_smc['high_volume_flag'].iloc[i]:s+=1;cr.append("Volume_Confirmation")
            if df_smc['close'].iloc[i]<df_smc['open'].iloc[i]:s+=1;cr.append("Bearish_Candle")
            if s>=4:ss.append({'timestamp':df_smc.index[i],'type':'SELL','price':df_smc['close'].iloc[i],'score':s,'reasons':cr,'stop_loss':self.calculate_stop_loss(df_smc,i,'sell'),'take_profit':self.calculate_take_profit(df_smc,i,'sell')})
        return ss
    def calculate_stop_loss(self,df,idx,st):cp=df['close'].iloc[idx];lld=df['low'].iloc[max(0,idx-self.lookback_period):idx];lhd=df['high'].iloc[max(0,idx-self.lookback_period):idx];return lld.min()*0.999 if st=='buy'and not lld.empty else(lhd.max()*1.001 if st=='sell'and not lhd.empty else(cp*0.98 if st=='buy'else cp*1.02))
    def calculate_take_profit(self,df,idx,st):cp=df['close'].iloc[idx];sl=self.calculate_stop_loss(df,idx,st);r=abs(cp-sl);return cp+(r*2)if st=='buy'else(cp-(r*2)if r!=0 else(cp*(0.96 if st=='sell'else 1.04)))
    def filter_signals_by_time(self,signals,max_signals_per_day=4):
        if not signals:return[];signals_df=pd.DataFrame(signals)
        if signals_df.empty:return[]
        if 'timestamp'not in signals_df.columns or'score'not in signals_df.columns:return signals # Or empty if strict
        try:signals_df['date']=pd.to_datetime(signals_df['timestamp']).dt.date
        except:return signals # Error in date conversion
        fs=[];
        for _,dsg in signals_df.groupby('date'):fs.extend(dsg.sort_values('score',ascending=False).head(max_signals_per_day).to_dict('records'))
        return fs
    def run_analysis(self):
        df_smc=self.calculate_smc_indicators();bs=self.generate_buy_signals(df_smc);ss=self.generate_sell_signals(df_smc)
        asigs=bs+ss;fs=[]
        if asigs:asigs.sort(key=lambda x:pd.to_datetime(x['timestamp']));fs=self.filter_signals_by_time(asigs)
        return fs,df_smc

# 4. AI Model Architecture
class SimpleTradingModelFinal(nn.Module): 
    def __init__(self, input_dim, hidden_dim, num_lstm_layers, lstm_dropout, fc_dropout):
        super(SimpleTradingModelFinal, self).__init__()
        effective_lstm_dropout = lstm_dropout if num_lstm_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_lstm_layers, batch_first=True, dropout=effective_lstm_dropout)
        self.attention_fc = nn.Linear(hidden_dim, hidden_dim); self.attention_tanh = nn.Tanh()
        self.attention_vector = nn.Linear(hidden_dim, 1, bias=False)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(fc_dropout), nn.Linear(hidden_dim // 2, 3) )
    def forward(self, x):
        lstm_out, _ = self.lstm(x); attn_fc_out = self.attention_fc(lstm_out)
        attn_tanh_out = self.attention_tanh(attn_fc_out); attn_energies = self.attention_vector(attn_tanh_out)
        attention_weights = torch.softmax(attn_energies, dim=1); context_vector = torch.sum(lstm_out * attention_weights, dim=1)
        return self.classifier(context_vector)

# --- Global Configurations & Constants ---
# This list MUST match the features the 'tuned_advanced_model.pth' was trained on.
# It's derived from your Cell 5 output after selecting numeric dtypes.
ALL_FEATURE_COLUMNS_LIST = ['rsi_14', 'macd', 'macd_signal', 'macd_diff', 'bb_hband', 'bb_lband', 'bb_width', 'atr_14', 'price_change_pct', 'hl_range_pct', 'oc_range_pct', 'volume_change_pct', 'smc_bullish_ob_near', 'smc_bearish_ob_near', 'smc_fvg_bullish_active', 'smc_fvg_bearish_active', 'smc_liquidity_buy_retested_strength', 'smc_liquidity_sell_retested_strength']
# Base SMC flags are also features, ensure they are in the list if your model used them:
# Example: 'swing_high_flag', 'swing_low_flag', 'high_volume_flag'
# If your all_feature_columns from Cell 5 included these, add them here.
# For now, assuming the 18 derived ones were the final set for scaling.
# Let's assume 'all_feature_columns' from Cell 5 was indeed the list of 18 derived + TAs + base flags that were numeric.
# The list provided in your previous outputs was:
# ['rsi_14', 'macd', 'macd_signal', 'macd_diff', 'bb_hband', 'bb_lband', 'bb_width', 'atr_14', 'price_change_pct', 'hl_range_pct', 'oc_range_pct', 'volume_change_pct', 'smc_bullish_ob_near', 'smc_bearish_ob_near', 'smc_fvg_bullish_active', 'smc_fvg_bearish_active', 'smc_liquidity_buy_retested_strength', 'smc_liquidity_sell_retested_strength']
# This is 18. Your SmartMoneyAnalyzer adds 'swing_high_flag', 'swing_low_flag', 'high_volume_flag'.
# These flags are boolean but get converted to int (0/1) by `astype(float)` then `StandardScaler`.
# So the `input_dim` of your model and the scaler expect these too.
# Let's use the exact list from your Cell 5 output. If it was 18, then it did not include the boolean flags.
# If it included them after they were converted to int/float for scaling, we add them.
# Based on your Cell 5 provided code, all_feature_columns = base_smc_flag_cols + ta_cols + engineered_smc_cols
# So, it should be 3 + 8 + 6 = 17 features if `ta_cols` has 8 distinct items. Let's list them out for clarity.
# ta_cols: ['rsi_14', 'macd', 'macd_signal', 'macd_diff', 'bb_hband', 'bb_lband', 'bb_width', 'atr_14', 'price_change_pct', 'hl_range_pct', 'oc_range_pct', 'volume_change_pct'] - This is 12
# base_smc_flag_cols = ['swing_high_flag', 'swing_low_flag', 'high_volume_flag'] - This is 3
# engineered_smc_cols = ['smc_bullish_ob_near', 'smc_bearish_ob_near', 'smc_fvg_bullish_active', 'smc_fvg_bearish_active', 'smc_liquidity_buy_retested_strength', 'smc_liquidity_sell_retested_strength'] - This is 6
# Total features that were scaled and model trained on = 12 + 3 + 6 = 21.
# The output "Processed 18 numeric features for TFT" might have been after some other filtering or for a different model.
# We need the features for `tuned_advanced_model.pth`.
# Let's assume the 21 features were used for the model being loaded.
ALL_FEATURE_COLUMNS_LIST = [
    'swing_high_flag', 'swing_low_flag', 'high_volume_flag', # Base SMC flags
    'rsi_14', 'macd', 'macd_signal', 'macd_diff', 'bb_hband', 'bb_lband', 'bb_width', 'atr_14', # TAs
    'price_change_pct', 'hl_range_pct', 'oc_range_pct', 'volume_change_pct', # Price/Volume TAs
    'smc_bullish_ob_near', 'smc_bearish_ob_near', 'smc_fvg_bullish_active', 'smc_fvg_bearish_active', # Derived SMC
    'smc_liquidity_buy_retested_strength', 'smc_liquidity_sell_retested_strength' # Derived SMC
]


MODEL_TO_USE_PATH = 'tuned_advanced_model.pth' 
SCALER_TO_USE_PATH = 'advanced_trading_scaler.pkl'
CHOSEN_MODEL_ARCHITECTURE_PARAMS = {
    'input_dim': len(ALL_FEATURE_COLUMNS_LIST), # Should be 21 now
    'hidden_dim': 128, 'num_lstm_layers': 1, 'lstm_dropout': 0.45, 'fc_dropout': 0.30 
}
SEQUENCE_LENGTH_FOR_AI = 20 
AI_CONFIDENCE_THRESHOLD = 0.65 
COMBINED_SIGNAL_SCORE_THRESHOLD = 6.0 

# Global instance of KrakenDataFetcher
kraken_fetcher_instance = None
try:
    kraken_fetcher_instance = KrakenDataFetcher()
    print("✓ KrakenDataFetcher instance created globally.")
except Exception as e:
    print(f"✗ Failed to create KrakenDataFetcher instance: {e}")
    BOT_LOGIC_CAN_PROCEED = False # Cannot proceed without data fetcher

# --- Load AI Model and Scaler ---
loaded_ai_model = None; fitted_scaler = None
if BOT_LOGIC_CAN_PROCEED:
    # ... (Model and scaler loading logic - ensure paths are correct) ...
    if os.path.exists(MODEL_TO_USE_PATH) and os.path.exists(SCALER_TO_USE_PATH):
        print(f"  Loading AI model: {MODEL_TO_USE_PATH}"); 
        # Update input_dim here based on the actual length of ALL_FEATURE_COLUMNS_LIST
        CHOSEN_MODEL_ARCHITECTURE_PARAMS['input_dim'] = len(ALL_FEATURE_COLUMNS_LIST)
        loaded_ai_model = SimpleTradingModelFinal(**CHOSEN_MODEL_ARCHITECTURE_PARAMS)
        try: 
            loaded_ai_model.load_state_dict(torch.load(MODEL_TO_USE_PATH, map_location=torch.device('cpu')))
            loaded_ai_model.eval(); print("  ✓ AI Model loaded successfully.")
        except Exception as e: print(f"  ✗ Error loading AI model: {e}"); loaded_ai_model=None; BOT_LOGIC_CAN_PROCEED=False
        
        print(f"  Loading scaler: {SCALER_TO_USE_PATH}")
        try:
            with open(SCALER_TO_USE_PATH, 'rb') as f: fitted_scaler = pickle.load(f)
            print("  ✓ Scaler loaded successfully.")
            if hasattr(fitted_scaler, 'n_features_in_') and fitted_scaler.n_features_in_ != len(ALL_FEATURE_COLUMNS_LIST):
                print(f"WARNING: Scaler expected {fitted_scaler.n_features_in_} features, but ALL_FEATURE_COLUMNS_LIST has {len(ALL_FEATURE_COLUMNS_LIST)}.")
                # BOT_LOGIC_CAN_PROCEED = False # Optionally stop if mismatch
        except Exception as e: print(f"  ✗ Error loading scaler: {e}"); fitted_scaler=None; BOT_LOGIC_CAN_PROCEED=False
    else: 
        print(f"  ✗ Model ('{MODEL_TO_USE_PATH}') or Scaler ('{SCALER_TO_USE_PATH}') file not found."); BOT_LOGIC_CAN_PROCEED=False


# --- Telegram Setup ---
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
telegram_bot_client = None
if TELEGRAM_AVAILABLE and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
    try: telegram_bot_client = Bot(token=TELEGRAM_BOT_TOKEN); print("✓ Telegram client initialized.")
    except Exception as e: print(f"✗ Error initializing Telegram client: {e}"); TELEGRAM_AVAILABLE = False
else:
    print("Telegram credentials not found in environment or library missing. Alerts will be printed to console.")
    TELEGRAM_AVAILABLE = False # Ensure it's False if credentials missing

async def send_telegram_alert(bot_client, chat_id, message_text): # Defined once
    # ... (keep the send_telegram_alert function from previous full Cell 13) ...
    if not bot_client or not TELEGRAM_AVAILABLE:
        print("\n--- SIMULATED TELEGRAM ALERT ---\n" + message_text + "\n----------------------------------")
        return False
    try:
        await bot_client.send_message(chat_id=chat_id, text=str(message_text), parse_mode=ParseMode.MARKDOWN_V2)
        print(f"✓ Telegram alert sent to chat ID {chat_id}.")
        return True
    except Exception as e:
        print(f"✗ Error sending Telegram alert: {e}\n--- FAILED ALERT (Printed) ---\n{message_text}\n{'-'*30}")
        return False
        
def escape_markdown_v2(text_to_escape): # Defined once
    # ... (keep the escape_markdown_v2 function from previous full Cell 13) ...
    if text_to_escape is None: text_to_escape = 'N/A'
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return "".join(['\\' + char if char in escape_chars else str(char) for char in str(text_to_escape)])


# --- Main Signal Generation Pipeline ---
async def run_signal_cycle_and_alert(): 
    global BOT_LOGIC_CAN_PROCEED, kraken_fetcher_instance # Ensure kraken_fetcher_instance is accessible
    if not (BOT_LOGIC_CAN_PROCEED and loaded_ai_model and fitted_scaler and kraken_fetcher_instance):
        print("\n--- Bot Logic Skipped: Critical Prerequisites Missing (model, scaler, or data fetcher) ---")
        return

    print("\n--- Starting Bot Signal Generation Cycle ---")
    print("  Fetching live market data...")
    current_market_data = kraken_fetcher_instance.get_ohlc_data(interval_minutes=15)
    if current_market_data is None or current_market_data.empty or len(current_market_data) < 250:
        msg = f"Failed to fetch sufficient market data (got {len(current_market_data) if current_market_data is not None else 0}). Skipping cycle."
        print(f"  {msg}")
        if TELEGRAM_AVAILABLE and telegram_bot_client: await send_telegram_alert(telegram_bot_client, TELEGRAM_CHAT_ID, escape_markdown_v2(f"⚠️ Bot Error: {msg}"))
        return
    data_for_this_cycle = current_market_data.tail(250).copy() 
    print(f"  Using market data of shape: {data_for_this_cycle.shape} for this cycle.")

    print("  Generating SMC rule-based signals...")
    live_smc_rules_system = SmartMoneyTradingSystem(data_for_this_cycle)
    smc_rules_signals_list, _ = live_smc_rules_system.run_analysis() 
    print(f"  Generated {len(smc_rules_signals_list)} SMC rule-based signals.")

    print("  Preparing data for AI model prediction...")
    ai_model_input_df = data_for_this_cycle.copy()
    try:
        # This must run on a df that can be modified without affecting df_from_smc_rules_indicators if it's used elsewhere
        live_smc_analyzer_for_ai = SmartMoneyAnalyzer(ai_model_input_df.copy()) # Use a copy for analyzer
        df_features_for_ai = live_smc_analyzer_for_ai.df # This df now has 'swing_high_flag', 'vol_avg' etc.
        
        # Add TAs
        # ... (Full TA calculation block as in previous Cell 13) ...
        df_features_for_ai['rsi_14']=ta.momentum.RSIIndicator(df_features_for_ai['close'],window=14,fillna=False).rsi(); macd_ai=ta.trend.MACD(df_features_for_ai['close'],fillna=False); df_features_for_ai['macd']=macd_ai.macd(); df_features_for_ai['macd_signal']=macd_ai.macd_signal(); df_features_for_ai['macd_diff']=macd_ai.macd_diff(); bb_ai=ta.volatility.BollingerBands(df_features_for_ai['close'],window=20,fillna=False); df_features_for_ai['bb_hband']=bb_ai.bollinger_hband(); df_features_for_ai['bb_lband']=bb_ai.bollinger_lband(); df_features_for_ai['bb_width']=bb_ai.bollinger_wband(); df_features_for_ai['atr_14']=ta.volatility.AverageTrueRange(df_features_for_ai['high'],df_features_for_ai['low'],df_features_for_ai['close'],window=14,fillna=False).average_true_range(); df_features_for_ai['price_change_pct']=df_features_for_ai['close'].pct_change(); df_features_for_ai['hl_range_pct']=(df_features_for_ai['high']-df_features_for_ai['low'])/(df_features_for_ai['close']+1e-9); df_features_for_ai['oc_range_pct']=(df_features_for_ai['close']-df_features_for_ai['open'])/(df_features_for_ai['open']+1e-9); df_features_for_ai['volume_change_pct']=df_features_for_ai['volume'].pct_change()
        
        # Engineer derived SMC features using live_smc_analyzer_for_ai's detected events
        # (Full SMC Event to Feature logic from your Cell 5, adapted for live events and df_features_for_ai)
        # ... (This is the same block as the previous full Cell 13 provided) ...
        df_features_for_ai['smc_bullish_ob_near']=0.0; df_features_for_ai['smc_bearish_ob_near']=0.0; df_features_for_ai['smc_fvg_bullish_active']=0.0; df_features_for_ai['smc_fvg_bearish_active']=0.0; df_features_for_ai['smc_liquidity_buy_retested_strength']=0.0; df_features_for_ai['smc_liquidity_sell_retested_strength']=0.0; lookback_smc_event_live=10
        live_detected_order_blocks = live_smc_analyzer_for_ai.detect_order_blocks(); live_detected_fvgs = live_smc_analyzer_for_ai.detect_fair_value_gaps(); live_detected_liquidity_pools = live_smc_analyzer_for_ai.detect_liquidity_pools()
        for ob_type, obs_list_key in [('bullish','bullish'),('bearish','bearish')]:
            col_name=f'smc_{ob_type}_ob_near'
            for ob_event in live_detected_order_blocks[obs_list_key]:
                event_time=ob_event['timestamp'];st_b=event_time;et_b=event_time+pd.Timedelta(minutes=15*lookback_smc_event_live)
                ptt_idx=df_features_for_ai.index[(df_features_for_ai.index>=st_b)&(df_features_for_ai.index<=et_b)]
                if not ptt_idx.empty:df_features_for_ai.loc[ptt_idx,col_name]=np.maximum(df_features_for_ai.loc[ptt_idx,col_name].fillna(0).values,ob_event.get('strength',0.0))
        for fvg_event in live_detected_fvgs:
            et=fvg_event['timestamp'];ftp=fvg_event.get('type','').split('_')[0]
            if not ftp:continue;col_name=f"smc_fvg_{ftp}_active"; 
            if col_name not in df_features_for_ai.columns:df_features_for_ai[col_name]=0.0
            st_b=et;et_b=et+pd.Timedelta(minutes=15*lookback_smc_event_live)
            ptt_idx=df_features_for_ai.index[(df_features_for_ai.index>=st_b)&(df_features_for_ai.index<=et_b)]
            if not ptt_idx.empty:df_features_for_ai.loc[ptt_idx,col_name]=np.maximum(df_features_for_ai.loc[ptt_idx,col_name].fillna(0).values,fvg_event.get('gap_percentage',0.0))
        for lp_event in live_detected_liquidity_pools:
            et=lp_event['timestamp'];lt=lp_event.get('type','')
            if 'buy_side' in lt:col_name="smc_liquidity_buy_retested_strength"
            elif 'sell_side' in lt:col_name="smc_liquidity_sell_retested_strength"
            else:continue
            if col_name not in df_features_for_ai.columns:df_features_for_ai[col_name]=0.0
            if et in df_features_for_ai.index:cv=df_features_for_ai.loc[et,col_name];cv=0.0 if pd.isna(cv)else cv;ns=lp_event.get('strength',0.0);df_features_for_ai.loc[et,col_name]=np.maximum(cv,ns)
        
        missing_final_check = [col for col in ALL_FEATURE_COLUMNS_LIST if col not in df_features_for_ai.columns]
        for col in missing_final_check: df_features_for_ai[col]=0.0
        if missing_final_check: print(f"    WARNING: Added missing AI feature cols for scaling: {missing_final_check}")
        df_features_for_ai[ALL_FEATURE_COLUMNS_LIST]=df_features_for_ai[ALL_FEATURE_COLUMNS_LIST].fillna(method='ffill').fillna(method='bfill').fillna(0)
    except Exception as e: print(f"    ✗ Error AI features: {e}"); import traceback; traceback.print_exc(); df_features_for_ai=None 

    ai_predictions_list=[]
    if df_features_for_ai is not None and len(df_features_for_ai)>=SEQUENCE_LENGTH_FOR_AI:
        # ... (AI prediction logic - scale, predict - as in previous Cell 13) ...
        latest_feats_ai=df_features_for_ai[ALL_FEATURE_COLUMNS_LIST].tail(SEQUENCE_LENGTH_FOR_AI).values # Use ALL_FEATURE_COLUMNS_LIST
        if np.isnan(latest_feats_ai).any():latest_feats_ai=np.nan_to_num(latest_feats_ai)
        try:scaled_latest_feats=fitted_scaler.transform(latest_feats_ai)
        except Exception as e:print(f"    ✗ Scaler error: {e}");scaled_latest_feats=None
        if scaled_latest_feats is not None:
            with torch.no_grad():
                ip_tensor=torch.FloatTensor(scaled_latest_feats).unsqueeze(0).to('cpu') 
                lgs=loaded_ai_model(ip_tensor);probs=torch.softmax(lgs,dim=1).squeeze().cpu().numpy()
                pc_idx=np.argmax(probs);conf=probs[pc_idx]
            s_map={0:'SELL',1:'HOLD',2:'BUY'};ai_s_type=s_map.get(pc_idx,'UNK_AI')
            ai_predictions_list.append({'timestamp':data_for_this_cycle.index[-1],'type':ai_s_type,'price':data_for_this_cycle['close'].iloc[-1],'ai_confidence':float(conf),'raw_probabilities':{s_map.get(i,str(i)):float(p) for i,p in enumerate(probs)}})
            print(f"  ✓ AI Model Prediction: {ai_s_type} (Conf: {conf:.2f}) for {ai_predictions_list[0]['timestamp']}")
    
    # --- Signal Combination & Telegram Sending (from previous corrected Cell 13) ---
    # ... (This part should be mostly okay now with 'final_score' handled) ...
    print("  Combining SMC and AI signals...")
    final_signals_to_send = [] # Initialize here
    # ... (Keep the exact combination logic from the previous fully working Cell 13) ...
    for smc_sig in smc_rules_signals_list: 
        enhanced_sig = smc_sig.copy(); enhanced_sig['signal_quality'] = 'SMC_RULE'; enhanced_sig['ai_confidence'] = 0.0; enhanced_sig['final_score'] = smc_sig.get('score',0); enhanced_sig.setdefault('reasons',[])
        if ai_predictions_list:
            ai_pred = ai_predictions_list[0]
            if ai_pred['type'] == smc_sig['type'] and ai_pred['type'] != 'HOLD': 
                enhanced_sig['final_score'] = smc_sig.get('score',5) + ai_pred['ai_confidence']*5; enhanced_sig['ai_confidence'] = ai_pred['ai_confidence']; enhanced_sig['signal_quality'] = 'ENHANCED_SMC'; enhanced_sig['reasons'].append(f"AI_Confirms_{ai_pred['ai_confidence']:.2f}")
            elif ai_pred['type'] != 'HOLD': enhanced_sig['reasons'].append(f"AI_Disagrees(AI:{ai_pred['type']})"); enhanced_sig['final_score'] = smc_sig.get('score',5)*0.5
        if enhanced_sig.get('final_score',0) >= COMBINED_SIGNAL_SCORE_THRESHOLD : final_signals_to_send.append(enhanced_sig)
    for ai_pred in ai_predictions_list: 
        if ai_pred['type'] != 'HOLD' and ai_pred['ai_confidence'] >= AI_CONFIDENCE_THRESHOLD:
            is_dupe = any(s['type']==ai_pred['type'] and abs((pd.to_datetime(s['timestamp'])-pd.to_datetime(ai_pred['timestamp'])).total_seconds())<600 for s in final_signals_to_send if 'ENHANCED_SMC' in s.get('signal_quality',''))
            if not is_dupe:
                atr_val_ai = df_features_for_ai['atr_14'].iloc[-1] if 'atr_14' in df_features_for_ai and pd.notna(df_features_for_ai['atr_14'].iloc[-1]) else ai_pred['price']*0.01
                sl_price_ai,tp_price_ai=(ai_pred['price']-2*atr_val_ai,ai_pred['price']+3*atr_val_ai) if ai_pred['type']=='BUY' else (ai_pred['price']+2*atr_val_ai,ai_pred['price']-3*atr_val_ai)
                final_signals_to_send.append({'timestamp':ai_pred['timestamp'],'type':ai_pred['type'],'price':ai_pred['price'],'final_score':ai_pred['ai_confidence']*10,'signal_quality':'AI_ONLY','ai_confidence':ai_pred['ai_confidence'],'reasons':[f"HighAI_{ai_pred['ai_confidence']:.2f}"],'stop_loss':sl_price_ai,'take_profit':tp_price_ai})
    if final_signals_to_send:
        temp_df = pd.DataFrame(final_signals_to_send).sort_values('final_score',ascending=False)
        final_signals_to_send = temp_df.drop_duplicates(subset=['timestamp','type'],keep='first').to_dict('records')
    
    print(f"\n--- Final Signals to Send (FULL Logic, TG Integrated) ---")
    if not final_signals_to_send: 
        no_signal_msg="Bot cycle ran, no new combined signals met criteria."
        print(f"  {no_signal_msg}")
        # await send_telegram_alert(telegram_bot_client, TELEGRAM_CHAT_ID, escape_markdown_v2(no_signal_msg)) # Optional status
    else:
        # Filter for only latest candle signals before sending
        latest_candle_timestamp_in_data = data_for_this_cycle.index[-1]
        signals_for_current_alert = []
        for signal_candidate in final_signals_to_send:
            signal_timestamp = pd.to_datetime(signal_candidate['timestamp'])
            if latest_candle_timestamp_in_data - timedelta(minutes=30) <= signal_timestamp <= latest_candle_timestamp_in_data:
                 signals_for_current_alert.append(signal_candidate)
        
        if not signals_for_current_alert:
            print("  No signals generated for the latest candle(s) after final filtering.")
        else:
            print(f"  Sending {len(signals_for_current_alert)} signal(s) for the latest candle(s):")
            for signal_to_send in signals_for_current_alert: # Iterate through filtered list
                # ... (Keep your message formatting and await send_telegram_alert logic) ...
                sig_ts_str=pd.to_datetime(signal_to_send['timestamp']).strftime('%Y-%m-%d %H:%M:%S'); emoji="🟢" if signal_to_send['type']=="BUY" else "🔴"; price_s,sl_s,tp_s=signal_to_send.get('price',0.0),signal_to_send.get('stop_loss',0.0),signal_to_send.get('take_profit',0.0); final_score_s,ai_conf_s=signal_to_send.get('final_score',0.0),signal_to_send.get('ai_confidence',0.0); quality_s,reasons_list_s=signal_to_send.get('signal_quality','N/A'),signal_to_send.get('reasons',['N/A']); reasons_s=', '.join(reasons_list_s); risk_pct_s,reward_pct_s,rr_s=0.0,0.0,0.0;
                if price_s>0 and sl_s!=0 and tp_s!=0:risk_val_s=abs(price_s-sl_s);reward_val_s=abs(tp_s-price_s);risk_pct_s=(risk_val_s/price_s)*100 if price_s!=0 else 0;reward_pct_s=(reward_val_s/price_s)*100 if price_s!=0 else 0;rr_s=reward_pct_s/risk_pct_s if risk_pct_s>0 else 0.0
                message_text=(f"{emoji} *{escape_markdown_v2(signal_to_send['type'])} SIGNAL* {emoji}\n\n💰 *Entry*: ${escape_markdown_v2(f'{price_s:,.2f}')}\n⏰ *Time*: {escape_markdown_v2(sig_ts_str)}\n📊 *Score*: {escape_markdown_v2(f'{final_score_s:.1f}')}\n🤖 *AI Conf*: {escape_markdown_v2(f'{ai_conf_s:.2f}')}\n🏷️ *Quality*: {escape_markdown_v2(quality_s)}\n\n🛡️ *Risk Management*:\n🛑 *SL*: ${escape_markdown_v2(f'{sl_s:,.2f}')}\n🎯 *TP*: ${escape_markdown_v2(f'{tp_s:,.2f}')}\n📉 *Risk*: {escape_markdown_v2(f'{risk_pct_s:.2f}%')}\n📈 *Reward*: {escape_markdown_v2(f'{reward_pct_s:.2f}%')}\n💎 *R:R*: 1:{escape_markdown_v2(f'{rr_s:.1f}')}\n\n🔍 *Reasons*: {escape_markdown_v2(reasons_s)}\n\n⚡ _Sys: SMC\\+AI vReal_")
                await send_telegram_alert(telegram_bot_client, TELEGRAM_CHAT_ID, message_text)
                if TELEGRAM_AVAILABLE and telegram_bot_client: await asyncio.sleep(1)


async def main_colab_run(): # Renamed to avoid conflict if script is run directly
    if BOT_LOGIC_CAN_PROCEED: await run_signal_cycle_and_alert()
    else: print("Prerequisites not met. Skipping signal cycle.")

# --- Main Execution Block for Standalone Script ---
if __name__ == "__main__":
    print("Executing trading_bot.py script...")
    
    # Define ALL_FEATURE_COLUMNS_LIST here if not already defined from Colab context
    # This is crucial for the script to know what features the model expects.
    if 'ALL_FEATURE_COLUMNS_LIST' not in globals():
        ALL_FEATURE_COLUMNS_LIST = [ # This MUST match the training
            'swing_high_flag', 'swing_low_flag', 'high_volume_flag', 
            'rsi_14', 'macd', 'macd_signal', 'macd_diff', 'bb_hband', 'bb_lband', 'bb_width', 'atr_14', 
            'price_change_pct', 'hl_range_pct', 'oc_range_pct', 'volume_change_pct', 
            'smc_bullish_ob_near', 'smc_bearish_ob_near', 'smc_fvg_bullish_active', 'smc_fvg_bearish_active', 
            'smc_liquidity_buy_retested_strength', 'smc_liquidity_sell_retested_strength'
        ]
        print(f"Defined ALL_FEATURE_COLUMNS_LIST with {len(ALL_FEATURE_COLUMNS_LIST)} features for standalone run.")

    # Update CHOSEN_MODEL_ARCHITECTURE_PARAMS to use this list
    CHOSEN_MODEL_ARCHITECTURE_PARAMS['input_dim'] = len(ALL_FEATURE_COLUMNS_LIST)


    # Instantiate KrakenDataFetcher globally for the script
    try:
        kraken_fetcher_instance = KrakenDataFetcher()
        print("✓ KrakenDataFetcher instance created for script.")
    except Exception as e:
        print(f"✗ CRITICAL: Failed to create KrakenDataFetcher instance in script: {e}")
        BOT_LOGIC_CAN_PROCEED = False # Must stop if no data fetcher

    # Re-check BOT_LOGIC_CAN_PROCEED after attempting to init KrakenDataFetcher
    # and after all_feature_columns list is defined.
    # The checks for model/scaler files will happen inside run_signal_cycle_and_alert path.
    
    # Initial load of model/scaler to set BOT_LOGIC_CAN_PROCEED correctly for the first run
    if BOT_LOGIC_CAN_PROCEED:
        if os.path.exists(MODEL_TO_USE_PATH) and os.path.exists(SCALER_TO_USE_PATH):
            loaded_ai_model = SimpleTradingModelFinal(**CHOSEN_MODEL_ARCHITECTURE_PARAMS)
            try: 
                loaded_ai_model.load_state_dict(torch.load(MODEL_TO_USE_PATH, map_location=torch.device('cpu')))
                loaded_ai_model.eval()
            except Exception as e: print(f"  ✗ Error loading AI model in __main__: {e}"); BOT_LOGIC_CAN_PROCEED = False
            try:
                with open(SCALER_TO_USE_PATH, 'rb') as f: fitted_scaler = pickle.load(f)
            except Exception as e: print(f"  ✗ Error loading scaler in __main__: {e}"); BOT_LOGIC_CAN_PROCEED = False
        else: 
            print(f"  ✗ Model or Scaler file not found in __main__."); BOT_LOGIC_CAN_PROCEED = False


    if BOT_LOGIC_CAN_PROCEED:
        # For standalone script, you might want nest_asyncio if running in an env that needs it
        # However, for GitHub Actions, direct asyncio.run is usually fine.
        # For local testing, if you get "RuntimeError: asyncio.run() cannot be called from a running event loop",
        # then add nest_asyncio.
        # try:
        #     import nest_asyncio
        #     nest_asyncio.apply()
        # except ImportError:
        #     pass # nest_asyncio not installed, proceed without
            
        asyncio.run(main_colab_run()) # Changed main() to main_colab_run() to match definition
    else:
        print("BOT_LOGIC_CAN_PROCEED is False due to missing files, classes or undefined essential variables. Exiting script.")
