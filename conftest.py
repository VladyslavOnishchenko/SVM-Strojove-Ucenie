"""Konfigurácia pytest pre celý projekt."""
import sys
import os

# Pridá koreňový adresár projektu do sys.path
sys.path.insert(0, os.path.dirname(__file__))
