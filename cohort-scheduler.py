"""
Streamlit Cohort Scheduler with Customizable Prerequisites
Run with: streamlit run app.py
"""

import streamlit as st
from collections import defaultdict
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import csv
import io
import sys

# ========== SCHEDULER CLASSES ==========

@dataclass
class SchedulerConfig:
    max_terms: int = 100
    max_modules_per_cohort_per_term: int = 1
    verbose: bool = False

class CohortScheduler:
    def __init__(self, modules: List[str], prereqs: Dict[str, List[str]], 
                 cohort_starts: Dict[str, int], config: Optional[SchedulerConfig] = None):
        self.config = config or SchedulerConfig()
        self.modules = modules
        self.prereqs = prereqs
        self.cohort_starts = cohort_starts
        self.cohorts = list(cohort_starts.keys())
        self._reset_schedule()
    
    def _reset_schedule(self):
        self.schedule = defaultdict(lambda: defaultdict(list))
        self.cohort_progress = {c: {} for c in self.cohorts}
        self.module_runs = {m: [] for m in self.modules}
        self.cohort_last_active = {c: 0 for c in self.cohorts}
        self.cohort_module_count = {c: defaultdict(int) for c in self.cohorts}
    
    def can_take_module(self, cohort: str, module: str, term: int) -> bool:
        if term < self.cohort_starts[cohort]:
            return False
        if self.cohort_module_count[cohort][term] >= self.config.max_modules_per_cohort_per_term:
            return False
        if len(self.cohort_progress[cohort]) >= len(self.modules):
            return False
        for prereq in self.prereqs.get(module, []):
            if prereq not in self.cohort_progress[cohort]:
                return False
            if self.cohort_progress[cohort][prereq] >= term:
                return False
        if module in self.cohort_progress[cohort]:
            return False
        return True
    
    def schedule_module_run(self, module: str, term: int, cohorts: List[str]):
        self.schedule[term][module].extend(cohorts)
        for cohort in cohorts:
            self.cohort_progress[cohort][module] = term
            self.cohort_last_active[cohort] = max(self.cohort_last_active[cohort], term)
            self.cohort_module_count[cohort][term] += 1
        if term not in self.module_runs[module]:
            self.module_runs[module].append(term)
    
    def find_optimal_schedule(self) -> bool:
        self._reset_schedule()
        cohort_needs = {c: set(self.modules) for c in self.cohorts}
        
        for term in range(1, self.config.max_terms + 1):
            active_cohorts = [
                c for c in self.cohorts
                if term >= self.cohort_starts[c] 
                and cohort_needs[c]
                and self.cohort_module_count[c][term] < self.config.max_modules_per_cohort_per_term
            ]
            
            if not active_cohorts:
                continue
            
            scheduled_this_term = set()
            
            while active_cohorts:
                unscheduled = [c for c in active_cohorts if c not in scheduled_this_term]
                if not unscheduled:
                    break
                
                module_scores = []
                for module in self.modules:
                    if module not in cohort_needs[unscheduled[0]]:
                        continue
                    
                    eligible = [c for c in unscheduled if self.can_take_module(c, module, term)]
                    
                    if eligible:
                        score = len(eligible)
                        module_scores.append((score, module, eligible))
                
                if not module_scores:
                    break
                
                module_scores.sort(reverse=True, key=lambda x: x[0])
                _, best_module, eligible_cohorts = module_scores[0]
                
                self.schedule_module_run(best_module, term, eligible_cohorts)
                
                for cohort in eligible_cohorts:
                    cohort_needs[cohort].discard(best_module)
                    scheduled_this_term.add(cohort)
        
        return all(len(self.cohort_progress[c]) == len(self.modules) for c in self.cohorts)
    
    def get_schedule_summary(self) -> Dict[str, Any]:
        total_runs = sum(len(runs) for runs in self.module_runs.values())
        max_term = max(self.cohort_last_active.values()) if self.cohort_last_active else 0
        
        return {
            'total_runs': total_runs,
            'max_term': max_term,
            'module_runs': {m: len(runs) for m, runs in self.module_runs.items()},
            'cohort_progress': self.cohort_progress,
            'schedule': self.schedule
        }

# ========== HELPER FUNCTIONS ==========

def has_cycle(prereqs: Dict[str, List[str]], modules: List[str]) -> bool:
    graph = {m: prereqs.get(m, []) for m in modules}
    visited = set()
    rec_stack = set()

    def dfs(node: str) -> bool:
        visited.add(node)
        rec_stack.add(node)
        for neigh in graph[node]:
            if neigh not in visited:
                if dfs(neigh):
                    return True
            elif neigh in rec_stack:
                return True
        rec_stack.remove(node)
        return False

    for node in modules:
        if node not in visited:
            if dfs(node):
                return True
    return False

# ========== STREAMLIT APP ==========

st.set_page_config(page_title="Cohort Scheduler", page_icon="📚", layout="wide")

st.title("📚 Cohort Scheduler with Custom Prerequisites")
st.markdown("Configure module prerequisites and run the scheduling optimization.")

# Initialize session state
if 'prereqs' not in st.session_state:
    st.session_state.prereqs = {
        'M1': [], 'M2': [], 'M3': [], 'M4': [], 'M5': [], 'M6': [],
        'M7': [], 'M8': [], 'M9': [],
        'M10': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'],
        'M11': ['M10'],
        'M12': ['M11']
    }

# Module names
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

modules = [f'M{i}' for i in range(1, 13)]

# Sidebar for preset configurations
st.sidebar.header("⚙️ Configuration Presets")

preset = st.sidebar.selectbox(
    "Load Preset:",
    ["Custom", "No Prerequisites", "Phase-Based", "Sequential (Original)"]
)

if st.sidebar.button("Load Preset"):
    # Clear existing widget states to prevent conflicts
    for module in modules:
        # Clear multiselect keys
        multi_key = f"multi_{module}"
        if multi_key in st.session_state:
            del st.session_state[multi_key]
        
        # Clear checkbox keys
        for potential in modules:
            if potential != module:
                key = f"prereq_{module}_{potential}"
                if key in st.session_state:
                    del st.session_state[key]

    if preset == "No Prerequisites":
        st.session_state.prereqs = {
            'M1': [], 'M2': [], 'M3': [], 'M4': [], 'M5': [], 'M6': [],
            'M7': [], 'M8': [], 'M9': [],
            'M10': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'],
            'M11': ['M10'],
            'M12': ['M11']
        }
        st.sidebar.success("✅ No Prerequisites preset loaded!")
    
    elif preset == "Phase-Based":
        st.session_state.prereqs = {
            'M1': [], 'M2': [], 'M3': [], 'M4': [],
            'M5': ['M2', 'M3', 'M4'],
            'M6': ['M2', 'M3', 'M4'],
            'M7': ['M2', 'M3', 'M4'],
            'M8': ['M2', 'M3', 'M4'],
            'M9': ['M5', 'M6', 'M7', 'M8'],
            'M10': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'],
            'M11': ['M10'],
            'M12': ['M11']
        }
        st.sidebar.success("✅ Phase-Based preset loaded!")
    
    elif preset == "Sequential (Original)":
        st.session_state.prereqs = {
            'M1': [], 'M2': [], 'M3': [],
            'M4': ['M2', 'M3'],
            'M5': ['M4'],
            'M6': ['M5'],
            'M7': ['M6'],
            'M8': ['M6'],
            'M9': ['M5'],
            'M10': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'],
            'M11': ['M10'],
            'M12': ['M11']
        }
        st.sidebar.success("✅ Sequential preset loaded!")
    
    # Force a rerun to refresh the UI with new prerequisites
    st.rerun()

# Main content tabs
tab1, tab2, tab3 = st.tabs(["📝 Edit Prerequisites", "▶️ Run Scheduler", "📊 Results"])

with tab1:
    st.header("Configure Module Prerequisites")
    st.markdown("Select which modules are required before each module can be taken.")
    
    # Edit mode selector
    edit_mode = st.radio(
        "Edit Mode:",
        ["Table View (Recommended)", "Individual Module Editor"],
        horizontal=True
    )
    
    if edit_mode == "Table View (Recommended)":
        st.markdown("**Check the boxes to set prerequisites. Each column represents a prerequisite.**")
        
        # Create a matrix view
        col_headers = st.columns([2] + [1] * len(modules))
        col_headers[0].markdown("**Module**")
        for i, mod in enumerate(modules):
            col_headers[i+1].markdown(f"**{mod}**")
        
        # Create rows for each module
        for module in modules:
            cols = st.columns([2] + [1] * len(modules))
            cols[0].markdown(f"**{module}** - {module_names[module]}")
            
            # Checkboxes for each potential prerequisite
            for i, potential_prereq in enumerate(modules):
                if potential_prereq == module:
                    cols[i+1].markdown("—")
                else:
                    current_prereqs = st.session_state.prereqs.get(module, [])
                    is_checked = potential_prereq in current_prereqs
                    
                    checked = cols[i+1].checkbox(
                        "",
                        value=is_checked,
                        key=f"prereq_{module}_{potential_prereq}",
                        label_visibility="collapsed"
                    )
                    
                    # Update session state
                    if checked and potential_prereq not in current_prereqs:
                        st.session_state.prereqs[module].append(potential_prereq)
                    elif not checked and potential_prereq in current_prereqs:
                        st.session_state.prereqs[module].remove(potential_prereq)
    
    else:  # Individual Module Editor
        st.markdown("**Select prerequisites for each module individually.**")
        
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        
        for idx, module in enumerate(modules):
            with (col1 if idx < 6 else col2):
                st.subheader(f"{module} - {module_names[module]}")
                
                # Get available prerequisites (all modules except itself)
                available_prereqs = [m for m in modules if m != module]
                
                # Multiselect for prerequisites
                selected = st.multiselect(
                    f"Prerequisites for {module}:",
                    available_prereqs,
                    default=st.session_state.prereqs.get(module, []),
                    key=f"multi_{module}",
                    format_func=lambda x: f"{x} - {module_names[x]}"
                )
                
                st.session_state.prereqs[module] = selected
    
    # Display current configuration
    st.divider()
    st.subheader("Current Prerequisite Configuration")
    
    config_display = []
    for module in modules:
        prereq_list = st.session_state.prereqs.get(module, [])
        if prereq_list:
            prereq_names = [f"{p} ({module_names[p]})" for p in prereq_list]
            config_display.append(f"**{module} ({module_names[module]}):** {', '.join(prereq_names)}")
        else:
            config_display.append(f"**{module} ({module_names[module]}):** No prerequisites")
    
    st.markdown("\n\n".join(config_display))

with tab2:
    st.header("Run Scheduler")
    
    # Configuration options
    col1, col2 = st.columns(2)
    with col1:
        max_terms = st.number_input("Maximum Terms", min_value=10, max_value=200, value=50)
    with col2:
        modules_per_term = st.number_input("Max Modules per Cohort per Term", min_value=1, max_value=5, value=1)
    
    if st.button("🚀 Run Scheduler", type="primary", use_container_width=True):
        with st.spinner("Running optimization..."):
            # Capture output
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            try:
                # Check for cycles
                if has_cycle(st.session_state.prereqs, modules):
                    st.error("❌ Cycle detected in prerequisite graph! Please fix the circular dependencies.")
                    sys.stdout = old_stdout
                else:
                    # Set up cohort configuration
                    cohort_starts = {
                        'C1': 1, 'C2': 2, 'C3': 4, 'C4': 6,
                        'C5': 8, 'C6': 10, 'C7': 12, 'C8': 14
                    }
                    
                    # Run scheduler
                    config = SchedulerConfig(
                        max_terms=max_terms,
                        max_modules_per_cohort_per_term=modules_per_term,
                        verbose=False
                    )
                    
                    optimized = CohortScheduler(
                        modules=modules,
                        prereqs=st.session_state.prereqs,
                        cohort_starts=cohort_starts,
                        config=config
                    )
                    
                    success = optimized.find_optimal_schedule()
                    summary = optimized.get_schedule_summary()
                    
                    # Store results in session state
                    st.session_state.scheduler = optimized
                    st.session_state.summary = summary
                    st.session_state.success = success
                    
                    output = sys.stdout.getvalue()
                    sys.stdout = old_stdout
                    
                    if success:
                        st.success("✅ Scheduling completed successfully!")
                    else:
                        st.warning("⚠️ Scheduling completed but some cohorts may not have finished all modules.")
                    
                    # Show summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Module Runs", summary['total_runs'])
                    with col2:
                        st.metric("Schedule Length", f"{summary['max_term']} terms")
                    with col3:
                        baseline_runs = 8 * 12
                        improvement = (baseline_runs - summary['total_runs']) / baseline_runs * 100
                        st.metric("Improvement vs Baseline", f"{improvement:.1f}%")
                    
                    st.info("✨ Go to the 'Results' tab to view detailed schedule and download CSV files.")
                
            except Exception as e:
                sys.stdout = old_stdout
                st.error(f"❌ Error running scheduler: {str(e)}")
                st.exception(e)

with tab3:
    st.header("Results & Downloads")
    
    if 'scheduler' not in st.session_state:
        st.info("👈 Please run the scheduler first in the 'Run Scheduler' tab.")
    else:
        scheduler = st.session_state.scheduler
        summary = st.session_state.summary
        
        # Download buttons
        st.subheader("📥 Download CSV Files")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Generate CSVs
        def generate_term_csv():
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(['Term', 'Module_Code', 'Module_Name', 'Cohorts'])
            for term in sorted(scheduler.schedule.keys()):
                for module, cohorts in scheduler.schedule[term].items():
                    writer.writerow([term, module, module_names[module], ','.join(sorted(cohorts))])
            return output.getvalue()
        
        def generate_cohort_csv():
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(['Cohort', 'Module_Code', 'Module_Name', 'Term_Taken'])
            for cohort in sorted(scheduler.cohort_progress.keys()):
                for module, term in sorted(scheduler.cohort_progress[cohort].items(), key=lambda x: x[1]):
                    writer.writerow([cohort, module, module_names[module], term])
            return output.getvalue()
        
        def generate_metrics_csv():
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(['Metric', 'Baseline', 'Optimized', 'Improvement_Percent'])
            baseline_runs = 96
            opt_runs = summary['total_runs']
            improvement = (baseline_runs - opt_runs) / baseline_runs * 100
            writer.writerow(['Total_Module_Runs', baseline_runs, opt_runs, f"{improvement:.2f}"])
            return output.getvalue()
        
        def generate_mapping_csv():
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(['Module_Code', 'Module_Name', 'Terms'])
            for module in sorted(module_names.keys()):
                terms = sorted(scheduler.module_runs[module])
                writer.writerow([module, module_names[module], ','.join(map(str, terms))])
            return output.getvalue()
        
        with col1:
            st.download_button(
                "📄 Term Schedule",
                generate_term_csv(),
                "term_by_term_schedule.csv",
                "text/csv"
            )
        
        with col2:
            st.download_button(
                "📄 Cohort Progression",
                generate_cohort_csv(),
                "cohort_progression.csv",
                "text/csv"
            )
        
        with col3:
            st.download_button(
                "📄 Metrics",
                generate_metrics_csv(),
                "metrics_comparison.csv",
                "text/csv"
            )
        
        with col4:
            st.download_button(
                "📄 Module Mapping",
                generate_mapping_csv(),
                "module_term_mapping.csv",
                "text/csv"
            )
        
        # Display detailed results
        st.divider()
        
        # Term-by-term schedule
        st.subheader("📅 Term-by-Term Schedule")
        for term in sorted(scheduler.schedule.keys()):
            if scheduler.schedule[term]:
                with st.expander(f"Term {term}"):
                    for module, cohorts in sorted(scheduler.schedule[term].items()):
                        st.write(f"**{module_names[module]}** ({module}) → Cohorts: {', '.join(sorted(cohorts))}")
        
        # Cohort progression
        st.subheader("👥 Cohort Progression")
        for cohort in sorted(scheduler.cohort_progress.keys()):
            with st.expander(f"Cohort {cohort}"):
                cohort_modules = sorted(
                    scheduler.cohort_progress[cohort].items(),
                    key=lambda x: x[1]
                )
                for module, term in cohort_modules:
                    st.write(f"Term {term}: **{module_names[module]}** ({module})")
        
        # Module-term mapping
        st.subheader("🗺️ Module-Term Mapping")
        mapping_cols = st.columns(3)
        for idx, module in enumerate(sorted(module_names.keys())):
            with mapping_cols[idx % 3]:
                terms = sorted(scheduler.module_runs[module])
                if terms:
                    st.write(f"**{module_names[module]}**")
                    st.write(f"Terms: {', '.join(map(str, terms))}")
                    st.write(f"({len(terms)} runs)")
                    st.markdown("---")

# Footer
st.divider()
st.markdown("**💡 Tip:** Try different prerequisite configurations to see how they affect scheduling efficiency!")
