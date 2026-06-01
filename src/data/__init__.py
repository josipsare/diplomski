"""Data layer.

    sec_parser     — low-level reader for SEC EDGAR Financial Statement Data Sets
    sec_loader     — iterates over all quarterly archives and produces a long-format panel
    stock_loader   — loads daily price files and computes per-day return / volume series
    panel_builder  — assembles unified quarterly + annual panels for downstream use
"""
