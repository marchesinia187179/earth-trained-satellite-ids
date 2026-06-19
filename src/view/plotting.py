"""
Plotting utilities for Satellite IDS project.

All functions read CSV files with pandas and produce visualizations
using matplotlib and seaborn. Each function is robust to missing or
empty CSVs and uses a clean plotting style.

Functions provided:
- plot_confusion_matrix(csv_path)
- plot_feature_importance(csv_path)
- plot_classification_report(csv_path)
- plot_learning_curves_iterative(csv_path)

All user-facing text is in English.
"""
from pathlib import Path
import time
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


sns.set_theme(style="whitegrid")


def _safe_read_csv(csv_path: str) -> Optional[pd.DataFrame]:
	"""Attempt to read a CSV file and return a DataFrame or None.

	Handles missing files and empty data gracefully.
	"""
	path = Path(csv_path)
	try:
		if not path.exists():
			print(f"[plotting] File not found: {csv_path}")
			return None
		df = pd.read_csv(path)
		if df.empty:
			print(f"[plotting] CSV is empty: {csv_path}")
			return None
		return df
	except pd.errors.EmptyDataError:
		print(f"[plotting] EmptyDataError while reading: {csv_path}")
		return None
	except Exception as e:
		print(f"[plotting] Error reading CSV {csv_path}: {e}")
		return None


def plot_confusion_matrix(csv_path: str, cmap: str = "Blues", save_dir: Optional[str] = None):
	"""Read a CSV containing a confusion matrix or predictions and plot a heatmap.

	The function accepts either:
	- a square CSV where rows and columns represent true/predicted labels,
	  or
	- a two-column CSV with columns like ['y_true','y_pred'] (names tolerant).
	"""
	df = _safe_read_csv(csv_path)
	if df is None:
		return

	# Case 1: predictions table
	cols_lower = [c.lower() for c in df.columns]
	if len(df.columns) >= 2 and ("y_true" in cols_lower or "true" in cols_lower) and ("y_pred" in cols_lower or "pred" in cols_lower):
		# try to identify true/pred columns
		true_col = df.columns[[c.lower() in ("y_true", "true") for c in df.columns]][0]
		pred_col = df.columns[[c.lower() in ("y_pred", "pred") for c in df.columns]][0]
		cm = pd.crosstab(df[true_col], df[pred_col])
	else:
		# Case 2: assume CSV itself is the confusion matrix
		try:
			cm = df.set_index(df.columns[0]) if df.shape[0] == df.shape[1] else df
		except Exception:
			cm = df

	plt.figure(figsize=(6, 5))
	sns.heatmap(cm, annot=True, fmt="g", cmap=cmap)
	plt.title("Confusion Matrix")
	plt.xlabel("Predicted")
	plt.ylabel("True")
	fig = plt.gcf()
	plt.tight_layout()
	if save_dir:
		save_path = Path(save_dir)
		save_path.mkdir(parents=True, exist_ok=True)
		out_file = save_path / f"{Path(csv_path).stem}_confusion_matrix.png"
		fig.savefig(out_file)
		plt.close(fig)
		print(f"[plotting] Saved confusion matrix to {out_file}")
	else:
		plt.show()


def plot_feature_importance(csv_path: str, top_n: int = 15, save_dir: Optional[str] = None):
	"""Read a CSV with columns ['Feature','Importance'] and plot top N horizontal bars."""
	df = _safe_read_csv(csv_path)
	if df is None:
		return

	if 'Feature' not in df.columns and 'feature' in [c.lower() for c in df.columns]:
		# normalize column names
		df.columns = [c if c.lower() != 'feature' else 'Feature' for c in df.columns]

	if 'Importance' not in df.columns and 'importance' in [c.lower() for c in df.columns]:
		df.columns = [c if c.lower() != 'importance' else 'Importance' for c in df.columns]

	if 'Feature' not in df.columns or 'Importance' not in df.columns:
		print(f"[plotting] Expected columns ['Feature','Importance'] in {csv_path}")
		return

	df_sorted = df.copy()
	df_sorted = df_sorted.sort_values('Importance', ascending=False).head(top_n)

	plt.figure(figsize=(8, max(4, top_n * 0.25)))
	sns.barplot(x='Importance', y='Feature', data=df_sorted, palette='viridis')
	plt.title(f'Top {top_n} Feature Importances')
	plt.xlabel('Importance')
	plt.ylabel('Feature')
	fig = plt.gcf()
	plt.tight_layout()
	if save_dir:
		save_path = Path(save_dir)
		save_path.mkdir(parents=True, exist_ok=True)
		out_file = save_path / f"{Path(csv_path).stem}_feature_importance.png"
		fig.savefig(out_file)
		plt.close(fig)
		print(f"[plotting] Saved feature importance to {out_file}")
	else:
		plt.show()


def plot_classification_report(csv_path: str, cmap: str = 'rocket', save_dir: Optional[str] = None):
	"""Read a CSV classification report (per-class metrics) and plot a heatmap.

	Expects rows to correspond to class labels and columns to include metrics
	like precision, recall, f1-score (case-insensitive).
	"""
	df = _safe_read_csv(csv_path)
	if df is None:
		return

	# If first column is index (class labels) after read, try to set it
	if df.columns[0].lower() in ('class', 'label', 'labels'):
		df = df.set_index(df.columns[0])

	# select numeric metric columns
	numeric = df.select_dtypes(include=["number"])
	if numeric.empty:
		# attempt to coerce possible numeric-like columns
		for col in df.columns:
			try:
				df[col] = pd.to_numeric(df[col])
			except Exception:
				continue
		numeric = df.select_dtypes(include=["number"])

	if numeric.empty:
		print(f"[plotting] No numeric metric columns found in {csv_path}")
		return

	plt.figure(figsize=(max(6, numeric.shape[1] * 1.2), max(4, numeric.shape[0] * 0.4)))
	sns.heatmap(numeric, annot=True, fmt='.2f', cmap=cmap)
	plt.title('Classification Report Summary')
	plt.xlabel('Metrics')
	plt.ylabel('Classes')
	fig = plt.gcf()
	plt.tight_layout()
	if save_dir:
		save_path = Path(save_dir)
		save_path.mkdir(parents=True, exist_ok=True)
		out_file = save_path / f"{Path(csv_path).stem}_classification_report.png"
		fig.savefig(out_file)
		plt.close(fig)
		print(f"[plotting] Saved classification report to {out_file}")
	else:
		plt.show()


def plot_learning_curves_iterative(csv_path: str, poll_interval: float = 0.5, patience: int = 3, save_dir: Optional[str] = None):
	"""Dynamically plot learning curves by polling an incrementally-updated CSV file.

	The CSV should contain columns named 'Iteration' or 'Epoch' and numeric metric
	columns like 'Loss', 'Accuracy' or 'F1-Score'. The function will open the CSV,
	plot available metrics, and update the figure as new rows appear. It stops
	when the file has not changed for `patience` consecutive polls.
	"""
	path = Path(csv_path)
	if not path.exists():
		print(f"[plotting] File not found: {csv_path}")
		return

	plt.ion()
	fig, ax = plt.subplots(figsize=(8, 4))

	last_len = 0
	unchanged = 0

	try:
		while True:
			try:
				df = pd.read_csv(path)
			except Exception:
				# file may be being written; wait and retry
				time.sleep(poll_interval)
				continue

			if df is None or df.empty:
				time.sleep(poll_interval)
				continue

			# identify step column
			step_col = None
			for candidate in ('Iteration', 'Epoch'):
				if candidate in df.columns:
					step_col = candidate
					break
			if step_col is None:
				# fallback to index as step
				df = df.reset_index().rename(columns={'index': 'Step'})
				step_col = 'Step'

			metrics = [c for c in df.columns if c not in (step_col,)]
			# ensure numeric metrics
			numeric_metrics = []
			for m in metrics:
				try:
					df[m] = pd.to_numeric(df[m])
					numeric_metrics.append(m)
				except Exception:
					continue

			if not numeric_metrics:
				print(f"[plotting] No numeric metrics found in {csv_path}")
				return

			# redraw
			ax.clear()
			for m in numeric_metrics:
				ax.plot(df[step_col], df[m], marker='o', label=m)

			ax.set_xlabel(step_col)
			ax.set_ylabel('Value')
			ax.set_title('Learning Curves (iterative)')
			ax.legend()
			ax.grid(True)
			fig.canvas.draw()
			fig.canvas.flush_events()

			curr_len = len(df)
			if curr_len == last_len:
				unchanged += 1
			else:
				unchanged = 0
				last_len = curr_len

			if unchanged >= patience:
				# assume finished
				break

			time.sleep(poll_interval)

	except KeyboardInterrupt:
		print('[plotting] Iterative plotting interrupted by user.')
	finally:
		plt.ioff()
		fig = plt.gcf()
		plt.tight_layout()
		if save_dir:
			save_path = Path(save_dir)
			save_path.mkdir(parents=True, exist_ok=True)
			out_file = save_path / f"{Path(csv_path).stem}_learning_curves.png"
			fig.savefig(out_file)
			plt.close(fig)
			print(f"[plotting] Saved final learning curves to {out_file}")
		else:
			plt.show()

