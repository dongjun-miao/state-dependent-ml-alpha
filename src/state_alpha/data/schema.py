"""Versioned CRSP CIZ source and canonical schemas."""

from __future__ import annotations

from collections import OrderedDict

import pyarrow as pa


SCHEMA_VERSION = "crsp_ciz_canonical_v1"

SECURITY_SOURCE_TO_CANONICAL = OrderedDict(
    [
        ("PERMNO", "permno"),
        ("PERMCO", "permco"),
        ("Ticker", "ticker"),
        ("TradingSymbol", "trading_symbol"),
        ("SecInfoStartDt", "sec_info_start_date"),
        ("SecInfoEndDt", "sec_info_end_date"),
        ("SecurityBegDt", "security_begin_date"),
        ("SecurityEndDt", "security_end_date"),
        ("PrimaryExch", "primary_exchange"),
        ("ConditionalType", "conditional_type"),
        ("TradingStatusFlg", "trading_status"),
        ("SecurityType", "security_type"),
        ("SecuritySubType", "security_subtype"),
        ("ShareType", "share_type"),
        ("USIncFlg", "us_incorporated_flag"),
        ("IssuerType", "issuer_type"),
        ("SecurityActiveFlg", "security_active_flag"),
        ("SICCD", "siccd"),
        ("DlyCalDt", "date"),
        ("DlyDelFlg", "daily_delisting_flag"),
        ("DlyPrc", "price_raw"),
        ("DlyPrcFlg", "price_flag"),
        ("DlyCap", "market_cap_source"),
        ("DlyCapFlg", "market_cap_flag"),
        ("DlyPrevPrc", "previous_price_raw"),
        ("DlyPrevPrcFlg", "previous_price_flag"),
        ("DlyPrevDt", "previous_price_date"),
        ("DlyPrevCap", "previous_market_cap_source"),
        ("DlyPrevCapFlg", "previous_market_cap_flag"),
        ("DlyRet", "return_total"),
        ("DlyRetx", "return_ex_distribution"),
        ("DlyRetI", "return_index"),
        ("DlyRetMissFlg", "return_missing_flag"),
        ("DlyRetDurFlg", "return_duration_flag"),
        ("DlyFacPrc", "price_factor"),
        ("DlyVol", "volume"),
        ("DlyClose", "close_raw"),
        ("DlyLow", "low_raw"),
        ("DlyHigh", "high_raw"),
        ("DlyBid", "bid_raw"),
        ("DlyAsk", "ask_raw"),
        ("DlyOpen", "open_raw"),
        ("DlyNumTrd", "number_of_trades"),
        ("DlyPrcVol", "price_volume_source"),
        ("ShrOut", "shares_outstanding_source"),
        ("ShrSource", "shares_source"),
        ("DelActionType", "delisting_action_type"),
        ("DelStatusType", "delisting_status_type"),
        ("DelReasonType", "delisting_reason_type"),
        ("DelPaymentType", "delisting_payment_type"),
        ("vwretd", "market_vw_return"),
        ("vwretx", "market_vw_return_x"),
        ("ewretd", "market_ew_return"),
        ("ewretx", "market_ew_return_x"),
        ("sprtrn", "sp500_return"),
    ]
)

DISTRIBUTION_SOURCE_TO_CANONICAL = OrderedDict(
    [
        ("PERMNO", "permno"),
        ("PERMCO", "permco"),
        ("DlyCalDt", "date"),
        ("DisExDt", "distribution_ex_date"),
        ("DisSeqNbr", "distribution_sequence_number"),
        ("DisOrdinaryFlg", "distribution_ordinary_flag"),
        ("DisType", "distribution_type"),
        ("DisFreqType", "distribution_frequency_type"),
        ("DisPaymentType", "distribution_payment_type"),
        ("DisDetailType", "distribution_detail_type"),
        ("DisTaxType", "distribution_tax_type"),
        ("DisOrigCurType", "distribution_original_currency"),
        ("DisDivAmt", "distribution_dividend_amount"),
        ("DisFacPr", "distribution_price_factor"),
        ("DisFacShr", "distribution_share_factor"),
        ("DisDeclareDt", "distribution_declare_date"),
        ("DisRecordDt", "distribution_record_date"),
        ("DisPayDt", "distribution_pay_date"),
        ("DisPERMNO", "distribution_permno"),
        ("DisPERMCO", "distribution_permco"),
    ]
)

DISTRIBUTION_DETECTION_COLUMNS = (
    "DisExDt",
    "DisSeqNbr",
    "DisType",
    "DisDivAmt",
    "DisFacPr",
    "DisFacShr",
)

SOURCE_COLUMNS = list(
    OrderedDict.fromkeys(
        [
            *SECURITY_SOURCE_TO_CANONICAL,
            *DISTRIBUTION_SOURCE_TO_CANONICAL,
        ]
    )
)

SECURITY_DATE_FIELDS = {
    "sec_info_start_date",
    "sec_info_end_date",
    "security_begin_date",
    "security_end_date",
    "date",
    "previous_price_date",
}

SECURITY_INTEGER_FIELDS = {"permno", "permco", "siccd"}

SECURITY_FLOAT_FIELDS = {
    "price_raw",
    "market_cap_source",
    "previous_price_raw",
    "previous_market_cap_source",
    "return_total",
    "return_ex_distribution",
    "return_index",
    "price_factor",
    "volume",
    "close_raw",
    "low_raw",
    "high_raw",
    "bid_raw",
    "ask_raw",
    "open_raw",
    "number_of_trades",
    "price_volume_source",
    "shares_outstanding_source",
    "market_vw_return",
    "market_vw_return_x",
    "market_ew_return",
    "market_ew_return_x",
    "sp500_return",
}

DISTRIBUTION_DATE_FIELDS = {
    "date",
    "distribution_ex_date",
    "distribution_declare_date",
    "distribution_record_date",
    "distribution_pay_date",
}

DISTRIBUTION_INTEGER_FIELDS = {
    "permno",
    "permco",
    "distribution_sequence_number",
    "distribution_permno",
    "distribution_permco",
}

DISTRIBUTION_FLOAT_FIELDS = {
    "distribution_dividend_amount",
    "distribution_price_factor",
    "distribution_share_factor",
}


def _field(name: str, date_fields: set[str], int_fields: set[str], float_fields: set[str]) -> pa.Field:
    if name in date_fields:
        dtype = pa.date32()
    elif name in int_fields:
        dtype = pa.int64()
    elif name in float_fields:
        dtype = pa.float64()
    else:
        dtype = pa.string()
    return pa.field(name, dtype, nullable=True)


SECURITY_ARROW_SCHEMA = pa.schema(
    [
        *[
            _field(
                name,
                SECURITY_DATE_FIELDS,
                SECURITY_INTEGER_FIELDS,
                SECURITY_FLOAT_FIELDS,
            )
            for name in SECURITY_SOURCE_TO_CANONICAL.values()
        ],
        pa.field("source_file", pa.string(), nullable=False),
        pa.field("source_row_number", pa.int64(), nullable=False),
        pa.field("source_record_count", pa.int16(), nullable=False),
        pa.field("distribution_record_count", pa.int16(), nullable=False),
    ],
    metadata={b"schema_version": SCHEMA_VERSION.encode("utf-8")},
)

DISTRIBUTION_ARROW_SCHEMA = pa.schema(
    [
        *[
            _field(
                name,
                DISTRIBUTION_DATE_FIELDS,
                DISTRIBUTION_INTEGER_FIELDS,
                DISTRIBUTION_FLOAT_FIELDS,
            )
            for name in DISTRIBUTION_SOURCE_TO_CANONICAL.values()
        ],
        pa.field("source_file", pa.string(), nullable=False),
        pa.field("source_row_number", pa.int64(), nullable=False),
        pa.field("security_day_record_count", pa.int16(), nullable=False),
    ],
    metadata={b"schema_version": SCHEMA_VERSION.encode("utf-8")},
)

