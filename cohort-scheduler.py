#%%
"""
Cohort Scheduler - Original Case Study Focus
Features:
- Only runs the original 12-module case study
- Term-by-term output in requested format (T1: M1 (C1))
- Cohort progress in requested format (C1: M1(T1) M2(T2)...)
- Results saved to CSV files
- Module names updated to descriptive course titles
- Module-term mapping showing when each module runs
"""
# app.py
import streamlit as st
from io import StringIO
import sys

# ... [paste your entire script here] ...


from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
import csv

@dataclass
class SchedulerConfig:
    """Configuration for the scheduler"""
    max_terms: int = 100
    max_modules_per_cohort_per_term: int = 1  # Cohort capacity parameter
    verbose: bool = False

class CohortScheduler:
    """
    Optimizes scheduling of modules across cohorts to minimize total module runs
    while respecting all constraints.
    """
    
    def __init__(
        self,
        modules: List[str],
        prereqs: Dict[str, List[str]],
        cohort_starts: Dict[str, int],
        config: Optional[SchedulerConfig] = None
    ):
        self.config = config or SchedulerConfig()
        self.modules = modules
        self.prereqs = prereqs
        self.cohort_starts = cohort_starts
        self.cohorts = list(cohort_starts.keys())
        
        # Initialize data structures
        self._reset_schedule()
    
    def _reset_schedule(self):
        """Reset all scheduling data structures"""
        self.schedule: Dict[int, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.cohort_progress: Dict[str, Dict[str, int]] = {c: {} for c in self.cohorts}
        self.module_runs: Dict[str, List[int]] = {m: [] for m in self.modules}
        self.cohort_last_active: Dict[str, int] = {c: 0 for c in self.cohorts}
        self.cohort_module_count: Dict[str, Dict[int, int]] = {c: defaultdict(int) for c in self.cohorts}
    
    def can_take_module(self, cohort: str, module: str, term: int) -> bool:
        """Check if a cohort can take a module in a given term"""
        # Cohort must have started
        if term < self.cohort_starts[cohort]:
            return False
        
        # Check cohort capacity constraint
        if self.cohort_module_count[cohort][term] >= self.config.max_modules_per_cohort_per_term:
            return False
        
        # Cohort must not have completed all modules
        if len(self.cohort_progress[cohort]) >= len(self.modules):
            return False
        
        # Check prerequisites are satisfied
        for prereq in self.prereqs.get(module, []):
            if prereq not in self.cohort_progress[cohort]:
                return False
            # Prerequisite must have been taken in a previous term
            if self.cohort_progress[cohort][prereq] >= term:
                return False
        
        # Module must not have been taken already
        if module in self.cohort_progress[cohort]:
            return False
        
        return True
    
    def schedule_module_run(self, module: str, term: int, cohorts: List[str]):
        """Schedule a module run for specified cohorts in a term"""
        self.schedule[term][module].extend(cohorts)
        
        for cohort in cohorts:
            self.cohort_progress[cohort][module] = term
            self.cohort_last_active[cohort] = max(self.cohort_last_active[cohort], term)
            self.cohort_module_count[cohort][term] += 1
        
        if term not in self.module_runs[module]:
            self.module_runs[module].append(term)
    
    def find_optimal_schedule(self) -> bool:
        """
        Find an optimal schedule using an improved greedy algorithm.
        
        Strategy:
        1. For each term, identify cohorts that need modules
        2. Try to maximize cohorts per module run
        3. Prioritize modules that multiple cohorts can take together
        4. Avoid creating gaps in cohort schedules
        """
        self._reset_schedule()
        cohort_needs = {c: set(self.modules) for c in self.cohorts}
        
        for term in range(1, self.config.max_terms + 1):
            # Get active cohorts that need modules and have capacity
            active_cohorts = [
                c for c in self.cohorts
                if term >= self.cohort_starts[c] 
                and cohort_needs[c]
                and self.cohort_module_count[c][term] < self.config.max_modules_per_cohort_per_term
            ]
            
            if not active_cohorts:
                continue
            
            # Track which cohorts have been scheduled this term
            scheduled_this_term = set()
            
            # Find best module runs for this term
            while active_cohorts:
                unscheduled = [c for c in active_cohorts if c not in scheduled_this_term]
                if not unscheduled:
                    break
                
                # For each possible module, count eligible unscheduled cohorts
                module_scores = []
                for module in self.modules:
                    if module not in cohort_needs[unscheduled[0]]:
                        continue
                    
                    eligible = [
                        c for c in unscheduled
                        if self.can_take_module(c, module, term)
                    ]
                    
                    if eligible:
                        # Score: prioritize modules that many cohorts can take
                        score = len(eligible)
                        module_scores.append((score, module, eligible))
                
                if not module_scores:
                    break
                
                # Choose module with highest score
                module_scores.sort(reverse=True, key=lambda x: x[0])
                _, best_module, eligible_cohorts = module_scores[0]
                
                # Schedule this module run
                self.schedule_module_run(best_module, term, eligible_cohorts)
                
                # Update tracking
                for cohort in eligible_cohorts:
                    cohort_needs[cohort].discard(best_module)
                    scheduled_this_term.add(cohort)
        
        # Check if all cohorts completed all modules
        return all(len(self.cohort_progress[c]) == len(self.modules) for c in self.cohorts)
    
    def get_schedule_summary(self) -> Dict[str, Any]:
        """Get a summary of the schedule"""
        total_runs = sum(len(runs) for runs in self.module_runs.values())
        max_term = max(self.cohort_last_active.values()) if self.cohort_last_active else 0
        
        return {
            'total_runs': total_runs,
            'max_term': max_term,
            'module_runs': {m: len(runs) for m, runs in self.module_runs.items()},
            'cohort_progress': self.cohort_progress,
            'schedule': self.schedule
        }

class BaselineScheduler(CohortScheduler):
    """Baseline scheduler that schedules modules sequentially without optimization"""
    
    def find_optimal_schedule(self) -> bool:
        """Schedule each cohort independently in topological order"""
        self._reset_schedule()
        
        # Precompute topological order of modules
        topo_order = self._topological_sort()
        
        for cohort in self.cohorts:
            current_term = self.cohort_starts[cohort]
            taken = set()
            
            while len(taken) < len(self.modules):
                # Find available modules in topological order
                available = []
                for module in topo_order:
                    if module in taken:
                        continue
                    if all(p in taken for p in self.prereqs.get(module, [])):
                        available.append(module)
                
                if not available:
                    if self.config.verbose:
                        print(f"Warning: Cohort {cohort} has no available modules at term {current_term}")
                    current_term += 1
                    continue
                
                # Take up to capacity modules
                to_take = available[:self.config.max_modules_per_cohort_per_term]
                taken.update(to_take)
                
                # Schedule each module separately (no sharing)
                for module in to_take:
                    self.schedule_module_run(module, current_term, [cohort])
                
                current_term += 1
        
        return True
    
    def _topological_sort(self) -> List[str]:
        """Perform topological sort on module prerequisites"""
        # Build graph
        graph = {m: [] for m in self.modules}
        in_degree = {m: 0 for m in self.modules}
        
        for module, prereqs in self.prereqs.items():
            for p in prereqs:
                graph[p].append(module)
                in_degree[module] = in_degree.get(module, 0) + 1
        
        # Kahn's algorithm
        queue = [m for m in self.modules if in_degree.get(m, 0) == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            for neighbor in graph.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(result) != len(self.modules):
            raise ValueError("Prerequisite graph has a cycle")
        
        return result

def save_to_csv_files(optimized_scheduler, baseline_runs, opt_summary, cohort_starts, module_names):
    """Save schedule data to CSV files with descriptive module names"""
    
    # 1. Term-by-term schedule CSV
    with open('term_by_term_schedule.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Term', 'Module_Code', 'Module_Name', 'Cohorts'])
        
        max_term = opt_summary['max_term']
        for term in range(1, max_term + 1):
            if term in optimized_scheduler.schedule:
                for module, cohorts in optimized_scheduler.schedule[term].items():
                    writer.writerow([term, module, module_names[module], ','.join(sorted(cohorts))])
    
    # 2. Cohort progression CSV
    with open('cohort_progression.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Cohort', 'Module_Code', 'Module_Name', 'Term_Taken'])
        
        for cohort in sorted(optimized_scheduler.cohort_progress.keys()):
            cohort_modules = optimized_scheduler.cohort_progress[cohort]
            sorted_modules = sorted(cohort_modules.items(), key=lambda x: x[1])
            
            for module, term_taken in sorted_modules:
                writer.writerow([cohort, module, module_names[module], term_taken])
    
    # 3. Metrics comparison CSV
    with open('metrics_comparison.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Metric', 'Baseline', 'Optimized', 'Improvement_Percent'])
        
        baseline_max_term = 14 + 12 - 1  # 25
        opt_max_term = opt_summary['max_term']
        
        # Total runs metric
        opt_runs = opt_summary['total_runs']
        runs_improvement = (baseline_runs - opt_runs) / baseline_runs * 100
        writer.writerow(['Total_Module_Runs', baseline_runs, opt_runs, f"{runs_improvement:.2f}"])
        
        # Schedule length metric
        term_improvement = (baseline_max_term - opt_max_term) / baseline_max_term * 100
        writer.writerow(['Schedule_Length_Terms', baseline_max_term, opt_max_term, f"{term_improvement:.2f}"])
    
    # 4. Module-term mapping CSV
    with open('module_term_mapping.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Module_Code', 'Module_Name', 'Terms'])
        
        for module in sorted(module_names.keys()):
            terms = sorted(optimized_scheduler.module_runs[module])
            if terms:
                writer.writerow([module, module_names[module], ','.join(map(str, terms))])

def run_original_case_study():
    """Run the original case study with 12 modules for demonstration"""
    print("=" * 80)
    print("ORIGINAL CASE STUDY: 12 Modules, 8 Cohorts")
    print("=" * 80)
    
    # Module name mapping as requested
    module_names = {
        'M1': 'Research Methods',
        'M2': 'Programming & Algorithms',
        'M3': 'Data Programming',
        'M4': 'Artificial Intelligence',
        'M5': 'Machine Learning',
        'M6': 'Deep Learning',
        'M7': 'NLP',
        'M8': 'Computer Vision',
        'M9': 'MLOps',
        'M10': 'Final Project 1',
        'M11': 'Final Project 2',
        'M12': 'Final Project 3'
    }
    
    # Display module mapping
    print("\nMODULE MAPPING:")
    print("-" * 40)
    for code, name in module_names.items():
        print(f"{code}\t{name}")
    
    # Original configuration with descriptive names
    modules = [f'M{i}' for i in range(1, 13)]
    cohort_starts = {
        'C1': 1, 'C2': 2, 'C3': 4, 'C4': 6,
        'C5': 8, 'C6': 10, 'C7': 12, 'C8': 14
    }
    
    # Original prerequisites with descriptive names
    prereqs = {
        'M1' : [],
        'M2' : [],
        'M3' : [],
        'M4' : ['M2', 'M3'],
        'M5' : ['M4'],
        'M6' : ['M5'],
        'M7' : ['M6'],
        'M8' : ['M6'],
        'M9' : ['M5'],
        'M10': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'],
        'M11': ['M10'],
        'M12': ['M11']
    }
    #phase based
    prereqs = {
        'M1' : [],
        'M2' : [],
        'M3' : [],
        'M4' : [],
        'M5' : ['M2', 'M3', 'M4'],
        'M6' : ['M2', 'M3', 'M4'],#['M5'],
        'M7' : ['M2', 'M3', 'M4'],#['M6'],
        'M8' : ['M2', 'M3', 'M4'],#['M6'],
        'M9' : ['M5', 'M6', 'M7', 'M8'],
        'M10': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'],
        'M11': ['M10'],
        'M12': ['M11']
    }
    
    #no preq
    prereqs = {
        'M1' : [],
        'M2' : [],
        'M3' : [],
        'M4' : [],
        'M5' : [],#['M2', 'M3', 'M4'],
        'M6' : [],#['M2', 'M3', 'M4'],#['M5'],
        'M7' : [],#['M2', 'M3', 'M4'],#['M6'],
        'M8' : [],#['M2', 'M3', 'M4'],#['M6'],
        'M9' : [],#['M5', 'M6', 'M7', 'M8'],
        'M10': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'],
        'M11': ['M10'],
        'M12': ['M11']
    }    
    
    # Baseline scheduler (sequential)
    baseline_config = SchedulerConfig(
        max_terms=50,
        max_modules_per_cohort_per_term=1,
        verbose=False
    )
    
    print("\nRunning baseline scheduler (sequential)...")
    baseline = BaselineScheduler(
        modules=modules,
        prereqs=prereqs,
        cohort_starts=cohort_starts,
        config=baseline_config
    )
    baseline.find_optimal_schedule()
    baseline_summary = baseline.get_schedule_summary()
    
    print("\nRunning optimized scheduler...")
    opt_config = SchedulerConfig(
        max_terms=50,
        max_modules_per_cohort_per_term=1,
        verbose=False
    )
    optimized = CohortScheduler(
        modules=modules,
        prereqs=prereqs,
        cohort_starts=cohort_starts,
        config=opt_config
    )
    optimized.find_optimal_schedule()
    opt_summary = optimized.get_schedule_summary()
    
    # Print comparison
    print("\nRESULTS COMPARISON:")
    print("-" * 50)
    print(f"{'Metric':<25} {'Baseline':<15} {'Optimized':<15} {'Improvement':<15}")
    print("-" * 50)
    
    baseline_runs = 8 * 12  # 96
    opt_runs = opt_summary['total_runs']
    runs_improvement = (baseline_runs - opt_runs) / baseline_runs * 100
    
    baseline_max_term = 14 + 12 - 1  # 25
    opt_max_term = opt_summary['max_term']
    term_improvement = (baseline_max_term - opt_max_term) / baseline_max_term * 100
    
    print(f"{'Total Module Runs':<25} {baseline_runs:<15} {opt_runs:<15} {runs_improvement:.1f}%")
    print(f"{'Schedule Length (Terms)':<25} {baseline_max_term:<15} {opt_max_term:<15} {term_improvement:.1f}%")
    
    # Print term-by-term schedule in requested format with descriptive names
    print("\n" + "=" * 80)
    print("DETAILED OPTIMIZED SCHEDULE (Term-by-Term)")
    print("=" * 80)
    
    max_term = opt_summary['max_term']
    for term in range(1, max_term + 1):
        term_entries = []
        if term in optimized.schedule and optimized.schedule[term]:
            for module, cohorts in sorted(optimized.schedule[term].items()):
                module_name = module_names[module]
                cohorts_str = ', '.join(sorted(cohorts))
                term_entries.append(f"{module_name} ({cohorts_str})")
        
        if term_entries:
            print(f"T{term}: " + "\t".join(term_entries))
        else:
            print(f"T{term}: (No scheduled modules)")
    
    # Print per-cohort progression in requested format with descriptive names
    print("\n" + "=" * 80)
    print("PER-COHORT MODULE PROGRESSION")
    print("=" * 80)
    
    for cohort in sorted(cohort_starts.keys()):
        # Get all modules taken by this cohort with their terms
        cohort_modules = optimized.cohort_progress[cohort]
        sorted_modules = sorted(cohort_modules.items(), key=lambda x: x[1])
        
        # Format as requested: C1: M1(T1) M2(T2) ... with descriptive names
        progression_str = "\t".join([f"{module_names[module]}(T{term})" for module, term in sorted_modules])
        print(f"{cohort}: {progression_str}")
    
    # Print module-term mapping
    print("\n" + "=" * 80)
    print("MODULE-TERM MAPPING")
    print("=" * 80)
    
    for module in sorted(module_names.keys()):
        module_name = module_names[module]
        terms = sorted(optimized.module_runs[module])
        if terms:
            terms_str = ", ".join(f"T{term}" for term in terms)
            print(f"{module_name}: {terms_str}")
        else:
            print(f"{module_name}: (Not scheduled)")
    
    # Save results to CSV files
    print("\n" + "=" * 80)
    print("SAVING RESULTS TO CSV FILES")
    print("=" * 80)
    save_to_csv_files(optimized, baseline_runs, opt_summary, cohort_starts, module_names)
    print("Term-by-term schedule saved to: term_by_term_schedule.csv")
    print("Cohort progression saved to: cohort_progression.csv")
    print("Metrics comparison saved to: metrics_comparison.csv")
    print("Module-term mapping saved to: module_term_mapping.csv")

def main():
    """Main execution function - Only runs the original case study"""
    print("Cohort Scheduler - Original Case Study Focus")
    print("=" * 80)
    run_original_case_study()

if __name__ == "__main__":
    main()
    
st.title("Cohort Scheduler")

if st.button("Run Scheduler"):
    # Capture console output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    main()
    
    output = sys.stdout.getvalue()
    sys.stdout = old_stdout
    
    st.text(output)
    
    # Provide CSV downloads
    st.download_button("Download Term Schedule", 
                       open('term_by_term_schedule.csv').read(),
                       "term_by_term_schedule.csv")
    st.download_button("Download Cohort Progression", 
                       open('cohort_progression.csv').read(),
                       "cohort_progression.csv")
# %%
