from evaluator import eval_util
import matplotlib.pyplot as plt
import pandas as pd
from timelogging.timeLog import log
from typing import Dict, List

class EvalRunAnalyzer():
  def __init__(self, eval_path: str, eval_run_dir: str, metrics_for_comparison):
    self.eval_run_info = {}
    self.metrics = metrics_for_comparison
    self.eval_run_info['iterations'] = self.get_eval_run_iterations(eval_run_dir)
    self.df = pd.read_excel(eval_path + '/' + eval_run_dir + '/Overview.xlsx')
    self.analyze_metrics()

  def analyze_metrics(self):
    mean = self.extract_mean()
    std = self.extract_std()
    for metric in self.metrics:
      self.eval_run_info[metric + '_mean'] = mean[metric]
      self.eval_run_info[metric + '_std'] = std[metric]

  def extract_mean(self):
    mean = self.df.mean(skipna=True)
    return mean[1:]  # skips index

  def extract_std(self):
    std = self.df.std(skipna=True)
    return std[1:]  # skips index

  def get_eval_run_iterations(self, eval_run_dir: str) -> float:
    return float(eval_run_dir.split("-")[0])

  def get_run_info(self):
    return self.eval_run_info

class EvaluationComparer():
  def __init__(self, eval_path: str, metrics_for_comparison: List[str] = ['gold_score', 'fine_tuned_score', 'summary_similarity_score']):
    self.eval_path = eval_path
    self.metrics_for_comparison = metrics_for_comparison
    self.eval_dirs = eval_util.get_subdirs(self.eval_path)
    self.analysis_df = self.collect_evaluations()

  def get_eval_run_info(self, eval_run_dir: str) -> Dict[str, float]:
    eval_run_analyzer = EvalRunAnalyzer(self.eval_path, eval_run_dir, self.metrics_for_comparison)
    return eval_run_analyzer.get_run_info()

  def collect_evaluations(self):
    row_list = []
    for eval_dir in self.eval_dirs:
      eval_run_info = self.get_eval_run_info(eval_dir)
      row_list.append(eval_run_info)
    return convert_rows_to_sorted_df(row_list)

  def save_dataframe(self):
    self.analysis_table_path = self.eval_path + '/analysis.xlsx'
    self.analysis_df.to_excel(self.analysis_table_path)
    print(self.analysis_df)
    log("saved to", self.analysis_table_path)

  # def plot_analysis(self): TODO
  #   for _, row in self.analysis_df.iterrows():
  #     for metric in self.metrics_for_comparison:
  #       log(row)
  #       plt.plot(row['iterations'], row['fine_tuned_score_mean'])
  #   plt.show()
  #   plt.savefig(self.eval_path + '/plot.jpg')

def convert_rows_to_sorted_df(row_list):
  df = pd.DataFrame(row_list)
  return df.sort_values('iterations')
