# -*- coding: utf-8 -*-
"""
===================================
LongbridgeFetcher - 长桥证券数据源
===================================

数据来源：Longbridge OpenAPI（长桥证券）
特点：支持港股、美股、A股、期权等
优点：数据质量高、接口稳定、支持实时行情

流控策略：
1. 使用 Longbridge SDK 内置的流控机制
2. 实现指数退避重试
3. 支持异步请求模式

API 文档参考：
- https://open.longbridgeapp.com/docs
- https://github.com/longbridgeapp/openapi-sdk
"""

import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from .base import BaseFetcher, DataFetchError, RateLimitError, STANDARD_COLUMNS
from .realtime_types import (
    UnifiedRealtimeQuote, ChipDistribution, RealtimeSource,
    get_realtime_circuit_breaker, get_chip_circuit_breaker,
    safe_float, safe_int
)
from src.config import get_config

logger = logging.getLogger(__name__)


def _is_us_code(stock_code: str) -> bool:
    """
    判断代码是否为美股
    
    美股代码规则：
    - 1-5个大写字母，如 'AAPL' (苹果), 'TSLA' (特斯拉)
    - 可能包含 '.' 用于特殊股票类别，如 'BRK.B' (伯克希尔B类股)
    """
    import re
    code = stock_code.strip().upper()
    return bool(re.match(r'^[A-Z]{1,5}(\.[A-Z])?$', code))


def _is_hk_code(stock_code: str) -> bool:
    """
    判断代码是否为港股
    
    港股代码规则：
    - 5位数字代码，如 '00700' (腾讯控股)
    - 部分港股代码可能带有前缀，如 'hk00700', 'hk1810'
    """
    code = stock_code.lower()
    if code.startswith('hk'):
        numeric_part = code[2:]
        return numeric_part.isdigit() and 1 <= len(numeric_part) <= 5
    return code.isdigit() and len(code) == 5


def _is_etf_code(stock_code: str) -> bool:
    """
    判断代码是否为 ETF 基金
    
    ETF 代码规则：
    - 上交所 ETF: 51xxxx, 52xxxx, 56xxxx, 58xxxx
    - 深交所 ETF: 15xxxx, 16xxxx, 18xxxx
    """
    etf_prefixes = ('51', '52', '56', '58', '15', '16', '18')
    return stock_code.startswith(etf_prefixes) and len(stock_code) == 6


class LongbridgeFetcher(BaseFetcher):
    """
    Longbridge 数据源实现
    
    优先级：3（介于 Tushare 和 Baostock 之间）
    数据来源：Longbridge OpenAPI
    
    关键策略：
    - 使用 Longbridge SDK 获取实时行情和历史数据
    - 支持港股、美股、A股、ETF等多市场
    - 实现智能重试和熔断机制
    
    配置要求：
    - LONG_BRIDGE_APP_KEY: Longbridge App Key
    - LONG_BRIDGE_APP_SECRET: Longbridge App Secret
    - LONG_BRIDGE_ACCESS_TOKEN: Longbridge Access Token（可选，可通过 SDK 自动获取）
    """
    
    name = "LongbridgeFetcher"
    priority = 3  # 默认优先级
    
    def __init__(self):
        """
        初始化 LongbridgeFetcher
        
        根据配置动态调整优先级：
        - 如果配置了 Longbridge API Key，优先级提升为 1
        - 否则保持默认优先级 3（较低）
        """
        self._config = None
        self._is_initialized = False
        
        # 尝试初始化 Longbridge SDK
        self._init_client()
        
        # 根据初始化结果动态调整优先级
        self.priority = self._determine_priority()
    
    def _init_client(self) -> None:
        """
        初始化 Longbridge SDK 客户端
        
        支持两种配置方式：
        1. 直接配置 Access Token
        2. 通过 App Key + App Secret 自动获取 Token
        """
        config = get_config()
        
        # 检查是否配置了 Longbridge 相关参数
        has_config = (
            config.longbridge_app_key and config.longbridge_app_secret
        )
        
        if not has_config:
            logger.warning("Longbridge API Key 未配置，此数据源不可用")
            return
        
        try:
            import longbridge
            from longbridge.openapi import Config, QuoteContext
            
            # 获取配置参数
            app_key = config.longbridge_app_key
            app_secret = config.longbridge_app_secret
            access_token = config.longbridge_access_token
            
            logger.info(f"尝试初始化 Longbridge SDK，App Key: {app_key[:8]}...")
            
            # 创建配置
            lb_config = None
            if access_token:
                # 使用直接配置的 Access Token
                logger.info("使用配置的 Access Token")
                lb_config = Config(
                    app_key=app_key,
                    app_secret=app_secret,
                    access_token=access_token
                )
            else:
                # 使用 App Key + App Secret（SDK 会自动获取 Token）
                logger.info("使用 App Key + App Secret 自动获取 Token")
                lb_config = Config(
                    app_key=app_key,
                    app_secret=app_secret
                )
            
            # 保存配置
            self._config = lb_config
            self._is_initialized = True
            logger.info("Longbridge SDK 初始化成功")
            
            # 测试连接
            self._test_connection()
            
        except ImportError:
            logger.warning("未安装 longbridge 库，请运行: pip install longbridge")
        except Exception as e:
            logger.error(f"Longbridge SDK 初始化失败: {e}")
            self._config = None
            self._is_initialized = False
    
    def _test_connection(self) -> None:
        """
        测试 Longbridge 连接
        
        通过获取简单的市场状态来验证连接是否正常
        """
        try:
            # 尝试获取港股市场状态
            from longbridge.openapi import QuoteContext
            quote_ctx = QuoteContext(self._config)
            market_status = quote_ctx.trading_session()
            
            logger.debug(f"Longbridge 连接测试成功: {market_status}")
            
        except Exception as e:
            logger.warning(f"Longbridge 连接测试失败（可能不影响基本功能）: {e}")
    
    def _determine_priority(self) -> int:
        """
        根据配置和初始化状态确定优先级
        
        Returns:
            优先级数字（0=最高，数字越大优先级越低）
        """
        if self._is_initialized:
            # Longbridge 可用，提升为较高优先级
            logger.info("✅ Longbridge 数据源可用，优先级提升为 1")
            return 1
        else:
            # Longbridge 不可用，保持默认优先级
            return 3
    
    def is_available(self) -> bool:
        """
        检查数据源是否可用
        
        Returns:
            True 表示可用，False 表示不可用
        """
        return self._is_initialized and self._config is not None
    
    def _convert_stock_code(self, stock_code: str) -> str:
        """
        转换股票代码为 Longbridge 格式
        
        Longbridge 代码格式：
        - 港股：00700.HK
        - 美股：AAPL.US
        - A股：600519.SH（沪市）, 000001.SZ（深市）
        - ETF：512880.SH（沪市）, 159995.SZ（深市）
        
        Args:
            stock_code: 原始代码，如 '600519', '00700', 'AAPL'
            
        Returns:
            Longbridge 格式代码，如 '600519.SH', '00700.HK', 'AAPL.US'
        """
        code = stock_code.strip().upper()
        
        # 已经包含后缀的情况（如 '00700.HK'）
        if '.' in code:
            # 如果已经是后缀格式，直接返回
            return code
        
        # 根据代码类型判断市场
        if _is_hk_code(code):
            # 港股：5位数字，补零到5位，添加 .HK 后缀
            return f"{code.zfill(5)}.HK"
        elif _is_us_code(code):
            # 美股：1-5个字母，添加 .US 后缀
            return f"{code}.US"
        elif _is_etf_code(code):
            # ETF：根据代码前缀判断市场
            if code.startswith(('51', '52', '56', '58')):
                return f"{code}.SH"  # 上交所 ETF
            else:
                return f"{code}.SZ"  # 深交所 ETF
        else:
            # A股：根据代码前缀判断市场
            if code.startswith(('600', '601', '603', '688')):
                return f"{code}.SH"  # 沪市
            else:
                return f"{code}.SZ"  # 深市
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从 Longbridge 获取历史数据
        
        使用 Longbridge SDK 的 get_candlesticks 接口获取日线数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）
            
        Returns:
            历史数据 DataFrame
        """
        if not self.is_available():
            raise DataFetchError("Longbridge SDK 未初始化，请检查配置")
        
        # 转换代码格式
        lb_code = self._convert_stock_code(stock_code)
        
        logger.info(f"[Longbridge] 获取 {lb_code} 历史数据: {start_date} ~ {end_date}")
        
        try:
            import longbridge
            from longbridge.openapi import QuoteContext, Period, AdjustType
            
            # 创建 QuoteContext
            quote_ctx = QuoteContext(self._config)
            
            # 转换日期格式
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            # 获取日线数据
            candles = quote_ctx.history_candlesticks_by_date(
                symbol=lb_code,
                period=Period.Day,
                start=start_dt,
                end=end_dt,
                adjust_type=AdjustType.ForwardAdjust  # 前复权
            )
            
            # 转换为 DataFrame
            data = []
            for candle in candles:
                data.append({
                    'timestamp': candle.timestamp,
                    'open': candle.open,
                    'high': candle.high,
                    'low': candle.low,
                    'close': candle.close,
                    'volume': candle.volume,
                    'turnover': candle.turnover,
                })
            
            df = pd.DataFrame(data)
            
            if df.empty:
                logger.warning(f"[Longbridge] 未获取到 {lb_code} 的历史数据")
            else:
                logger.info(f"[Longbridge] 获取成功: {len(df)} 条数据，日期范围: {df.iloc[0]['timestamp']} ~ {df.iloc[-1]['timestamp']}")
            
            return df
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # 检测 API 限流或权限错误
            if any(keyword in error_msg for keyword in ['rate', 'limit', 'quota', 'permission', 'auth']):
                logger.warning(f"[Longbridge] 可能被限流或权限不足: {e}")
                raise RateLimitError(f"Longbridge API 限流: {e}") from e
            
            raise DataFetchError(f"Longbridge 获取历史数据失败: {e}") from e
    
    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """
        标准化 Longbridge 数据
        
        Longbridge 返回的数据结构：
        - timestamp: 时间戳
        - open: 开盘价
        - high: 最高价
        - low: 最低价
        - close: 收盘价
        - volume: 成交量（股）
        - turnover: 成交额（元）
        
        需要映射到标准列名：
        date, open, high, low, close, volume, amount, pct_chg
        """
        if df is None or df.empty:
            return pd.DataFrame(columns=['code'] + STANDARD_COLUMNS)
        
        df = df.copy()
        
        # 重命名列
        column_mapping = {
            'timestamp': 'date',
            'turnover': 'amount',
        }
        df = df.rename(columns=column_mapping)
        
        # 转换日期格式
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # 计算涨跌幅（如果数据包含多行）
        if len(df) > 1:
            df['pct_chg'] = df['close'].pct_change() * 100
            df.loc[0, 'pct_chg'] = 0  # 第一行设为0
        else:
            df['pct_chg'] = 0
        
        # 添加股票代码列
        df['code'] = stock_code
        
        # 只保留需要的列
        keep_cols = ['code'] + STANDARD_COLUMNS
        existing_cols = [col for col in keep_cols if col in df.columns]
        df = df[existing_cols]
        
        return df
    
    def get_realtime_quote(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        """
        获取实时行情数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            UnifiedRealtimeQuote 对象，获取失败返回 None
        """
        if not self.is_available():
            logger.debug(f"[Longbridge] 数据源不可用，跳过 {stock_code}")
            return None
        
        # 检查熔断器状态
        circuit_breaker = get_realtime_circuit_breaker()
        source_key = "longbridge"
        
        if not circuit_breaker.is_available(source_key):
            logger.warning(f"[熔断] Longbridge 处于熔断状态，跳过 {stock_code}")
            return None
        
        try:
            import longbridge
            from longbridge.openapi import QuoteContext
            
            # 转换代码格式
            lb_code = self._convert_stock_code(stock_code)
            
            # 创建 QuoteContext
            quote_ctx = QuoteContext(self._config)
            
            # 获取实时行情
            quotes = quote_ctx.quote([lb_code])
            if not quotes:
                logger.warning(f"[Longbridge] 未获取到 {stock_code} 的实时行情，返回空列表")
                circuit_breaker.record_failure(source_key, "返回空列表")
                return None
            quote = quotes[0]
            
            # 计算涨跌幅和涨跌额
            price = safe_float(quote.last_done)
            prev_close = safe_float(quote.prev_close)
            change_pct = None
            change_amount = None
            if price is not None and prev_close is not None and prev_close != 0:
                change_amount = price - prev_close
                change_pct = (change_amount / prev_close) * 100
            
            # 使用 realtime_types.py 中的统一转换函数
            # 注意：Longbridge 的 SecurityQuote 对象没有 symbol_name 属性，我们用股票代码代替名称
            realtime_quote = UnifiedRealtimeQuote(
                code=stock_code,
                name=stock_code,  # 暂时用股票代码代替，因为没有名称
                source=RealtimeSource.LONGBRIDGE,
                price=price,
                change_pct=change_pct,
                change_amount=change_amount,
                volume=safe_int(quote.volume),
                amount=safe_float(quote.turnover),
                open_price=safe_float(quote.open),
                high=safe_float(quote.high),
                low=safe_float(quote.low),
                pre_close=prev_close,  # 注意：字段名是 pre_close 不是 prev_close
                # 以下字段Longbridge的SecurityQuote未提供，使用默认值（None）
                # volume_ratio, turnover_rate, amplitude, pe_ratio, pb_ratio, total_mv, circ_mv, change_60d, high_52w, low_52w 均为可选字段，不传入则使用默认值None
            )
            
            circuit_breaker.record_success(source_key)
            
            logger.info(f"[Longbridge实时行情] {stock_code}: "
                       f"价格={realtime_quote.price}, 涨跌={realtime_quote.change_pct}%")
            
            return realtime_quote
            
        except Exception as e:
            logger.error(f"[Longbridge] 获取 {stock_code} 实时行情失败: {e}")
            circuit_breaker.record_failure(source_key, str(e))
            return None
    
    def get_chip_distribution(self, stock_code: str) -> Optional[ChipDistribution]:
        """
        获取筹码分布数据
        
        Longbridge 暂不直接提供筹码分布数据，返回 None
        
        Args:
            stock_code: 股票代码
            
        Returns:
            None（Longbridge 不提供此数据）
        """
        logger.debug(f"[Longbridge] 不提供筹码分布数据，跳过 {stock_code}")
        return None
    
    def get_enhanced_data(self, stock_code: str, days: int = 60) -> Dict[str, Any]:
        """
        获取增强数据（历史K线 + 实时行情）
        
        Args:
            stock_code: 股票代码
            days: 历史数据天数
            
        Returns:
            包含所有数据的字典
        """
        result = {
            'code': stock_code,
            'daily_data': None,
            'realtime_quote': None,
            'chip_distribution': None,
        }
        
        # 获取日线数据
        try:
            df = self.get_daily_data(stock_code, days=days)
            result['daily_data'] = df
        except Exception as e:
            logger.error(f"获取 {stock_code} 日线数据失败: {e}")
        
        # 获取实时行情
        result['realtime_quote'] = self.get_realtime_quote(stock_code)
        
        # 获取筹码分布（Longbridge 不提供）
        result['chip_distribution'] = self.get_chip_distribution(stock_code)
        
        return result


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.DEBUG)
    
    fetcher = LongbridgeFetcher()
    
    if fetcher.is_available():
        print("=" * 50)
        print("测试 Longbridge 数据源")
        print("=" * 50)
        
        # 测试港股
        print("\n测试港股历史数据获取:")
        try:
            df = fetcher.get_daily_data('00700')  # 腾讯控股
            print(f"[港股] 获取成功，共 {len(df)} 条数据")
            if not df.empty:
                print(df.tail())
        except Exception as e:
            print(f"[港股] 获取失败: {e}")
        
        # 测试美股
        print("\n测试美股历史数据获取:")
        try:
            df = fetcher.get_daily_data('AAPL')  # 苹果
            print(f"[美股] 获取成功，共 {len(df)} 条数据")
            if not df.empty:
                print(df.tail())
        except Exception as e:
            print(f"[美股] 获取失败: {e}")
        
        # 测试A股
        print("\n测试A股历史数据获取:")
        try:
            df = fetcher.get_daily_data('600519')  # 贵州茅台
            print(f"[A股] 获取成功，共 {len(df)} 条数据")
            if not df.empty:
                print(df.tail())
        except Exception as e:
            print(f"[A股] 获取失败: {e}")
        
        # 测试实时行情
        print("\n测试实时行情获取:")
        try:
            quote = fetcher.get_realtime_quote('00700')  # 腾讯控股
            if quote:
                print(f"[实时行情] {quote.name}: 价格={quote.price}, 涨跌幅={quote.change_pct}%")
            else:
                print("[实时行情] 未获取到数据")
        except Exception as e:
            print(f"[实时行情] 获取失败: {e}")
    else:
        print("Longbridge 数据源不可用，请检查配置")
