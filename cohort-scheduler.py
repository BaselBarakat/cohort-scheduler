"""
Streamlit Cohort Scheduler with Customizable Prerequisites
Run with: streamlit run app.py
"""

import streamlit as st
from collections import defaultdict
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
import csv
import io
import sys
import networkx as nx  # Added for dependency cycle detection

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
        """Check if a cohort can take a module in a given term"""
        # Basic checks
        if term < self.cohort_starts[cohort]:
            return False
        if self.cohort_module_count[cohort][term] >= self.config.max_modules_per_cohort_per_term:
            return False
        if module in self.cohort_progress[cohort]:
            return False
        
        # Check prerequisites
        for prereq in self.prereqs.get(module, []):
            # Skip invalid prerequisites not in our module list
            if prereq not in self.modules:
                continue
            if prereq not in self.cohort_progress[cohort]:
                return False
            if self.cohort_progress[cohort][prereq] >= term:
                return False
        return True
    
    def schedule_module_run(self, module: str, term: int, cohorts: List[str]):
        """Schedule a module run for multiple cohorts"""
        self.schedule[term][module].extend(cohorts)
        for cohort in cohorts:
            self.cohort_progress[cohort][module] = term
            self.cohort_last_active[cohort] = max(self.cohort_last_active[cohort], term)
            self.cohort_module_count[cohort][term] += 1
        if term not in self.module_runs[module]:
            self.module_runs[module].append(term)
    
    def detect_cycles(self) -> List[List[str]]:
        """Detect cycles in prerequisites using networkx"""
        G = nx.DiGraph()
        for module in self.modules:
            G.add_node(module)
            for prereq in self.prereqs.get(module, []):
                if prereq in self.modules:  # Only consider valid modules
                    G.add_edge(prereq, module)
        
        try:
            cycles = list(nx.simple_cycles(G))
            return cycles
        except nx.NetworkXError:
            return []
    
    def find_optimal_schedule(self) -> bool:
        """Find an optimal schedule for all cohorts"""
        # First check for dependency cycles
        cycles = self.detect_cycles()
        if cycles:
            error_msg = "❌ Found circular dependencies in prerequisites:\n"
            for cycle in cycles:
                error_msg += f"  → {' → '.join(cycle)} → {cycle[0]}\n"
            error_msg += "Please fix these cycles before scheduling."
            st.error(error_msg)
            return False
        
        self._reset_schedule()
        cohort_needs = {c: set(self.modules) for c in self.cohorts}
        
        # Track unscheduled modules to detect impossible schedules
        unscheduled_modules = set(self.modules)
        
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
            term_scheduled = False
            
            # Try to schedule modules for this term
            while active_cohorts:
                unscheduled = [c for c in active_cohorts if c not in scheduled_this_term]
                if not unscheduled:
                    break
                
                # Find best module to schedule
                module_scores = []
                for module in sorted(unscheduled_modules):  # Sort for deterministic behavior
                    if module not in cohort_needs[unscheduled[0]]:
                        continue
                    
                    eligible = [c for c in unscheduled if self.can_take_module(c, module, term)]
                    
                    if eligible:
                        score = len(eligible)
                        module_scores.append((score, module, eligible))
                
                if not module_scores:
                    break
                
                # Select module with most eligible cohorts
                module_scores.sort(reverse=True, key=lambda x: x[0])
                _, best_module, eligible_cohorts = module_scores[0]
                
                self.schedule_module_run(best_module, term, eligible_cohorts)
                term_scheduled = True
                
                # Update cohort needs
                for cohort in eligible_cohorts:
                    cohort_needs[cohort].discard(best_module)
                    scheduled_this_term.add(cohort)
                
                # Remove module from unscheduled if all cohorts can take it eventually
                if all(best_module not in cohort_needs[c] for c in self.cohorts):
                    unscheduled_modules.discard(best_module)
            
            # Early termination if nothing was scheduled this term
            if not term_scheduled:
                break
        
        # Check completion status
        all_modules_scheduled = all(len(self.cohort_progress[c]) == len(self.modules) for c in self.cohorts)
        
        # Handle incomplete schedules
        if not all_modules_scheduled:
            incomplete_cohorts = [c for c in self.cohorts if len(self.cohort_progress[c]) < len(self.modules)]
            missing_modules = {}
            for cohort in incomplete_cohorts:
                missing = set(self.modules) - set(self.cohort_progress[cohort].keys())
                missing_modules[cohort] = missing
            
            warning_msg = "⚠️ Schedule incomplete. The following cohorts couldn't finish all modules:\n"
            for cohort, modules in missing_modules.items():
                warning_msg += f"- Cohort {cohort}: Missing {', '.join(sorted(modules))}\n"
            warning_msg += "\nPossible causes:\n"
            warning_msg += "- Prerequisites create impossible sequences\n"
            warning_msg += "- Not enough terms allocated\n"
            warning_msg += "- Too few modules allowed per term"
            st.warning(warning_msg)
        
        return all_modules_scheduled
    
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

# ========== PRESET DEFINITIONS ==========

PRESETS = {
    "No Prerequisites": {
        'M1': [], 'M2': [], 'M3': [], 'M4': [], 'M5': [], 'M6': [],
        'M7': [], 'M8': [], 'M9': [],
        'M10': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'],
        'M11': ['M10'],
        'M12': ['M11']
    },
    "Phase-Based": {
        'M1': [], 'M2': [], 'M3': [], 'M4': [],
        'M5': ['M2', 'M3', 'M4'],
        'M6': ['M2', 'M3', 'M4'],
        'M7': ['M2', 'M3', 'M4'],
        'M8': ['M2', 'M3', 'M4'],
        'M9': ['M5', 'M6', 'M7', 'M8'],
        'M10': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'],
        'M11': ['M10'],
        'M12': ['M11']
    },
    "Sequential (Original)": {
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
}

# ========== STREAMLIT APP ==========

st.set_page_config(page_title="Cohort Scheduler", page_icon="📚", layout="wide")

st.title("📚 Cohort Scheduler with Custom Prerequisites")
st.markdown("Configure module prerequisites and run the scheduling optimization.")

# Module names mapping
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

# Initialize session state with proper defaults
if 'prereqs' not in st.session_state:
    st.session_state.prereqs = PRESETS["No Prerequisites"].copy()

# Ensure all modules exist in prerequisites
for module in modules:
    if module not in st.session_state.prereqs:
        st.session_state.prereqs[module] = []

# Sidebar for preset configurations
st.sidebar.header("⚙️ Configuration Presets")

preset = st.sidebar.selectbox(
    "Load Preset:",
    ["Custom", "No Prerequisites", "Phase-Based", "Sequential (Original)"]
)

# Reset to initial state button
if st.sidebar.button("↩️ Reset to Initial"):
    st.session_state.prereqs = PRESETS["No Prerequisites"].copy()
    st.success("✅ Reset to initial configuration!")
    st.rerun()

# Load preset button
if st.sidebar.button("✅ Load Preset"):
    if preset in PRESETS:
        st.session_state.prereqs = PRESETS[preset].copy()
        # Ensure all modules exist after loading preset
        for module in modules:
            if module not in st.session_state.prereqs:
                st.session_state.prereqs[module] = []
        st.sidebar.success(f"✅ {preset} preset loaded!")
    else:
        st.sidebar.warning("Custom configuration preserved")

# Main content tabs
tab1, tab2, tab3 = st.tabs(["📝 Edit Prerequisites", "▶️ Run Scheduler", "📊 Results"])

with tab1:
    st.header("Configure Module Prerequisites")
    st.markdown("Select which modules are required before each module can be taken.")
    
    # Edit mode selector
    edit_mode = st.radio(
        "Edit Mode:",
        ["Individual Module Editor (Recommended)", "Table View"],
        horizontal=True
    )
    
    if edit_mode == "Table View":
        st.markdown("**Check the boxes to set prerequisites. Rows = modules, Columns = prerequisites**")
        st.caption("💡 Tip: Use the individual editor for complex configurations")
        
        # Create scrollable container for the large table
        with st.container():
            # Column headers
            header_cols = st.columns([2.5] + [0.8] * len(modules))
            header_cols[0].markdown("**Module**")
            for i, mod in enumerate(modules):
                header_cols[i+1].markdown(f"**{mod}**", help=module_names[mod])
            
            # Create rows for each module with scrollable container
            for module in modules:
                cols = st.columns([2.5] + [0.8] * len(modules))
                cols[0].markdown(f"**{module}** - {module_names[module]}")
                
                # Checkboxes for each potential prerequisite
                for i, potential_prereq in enumerate(modules):
                    if potential_prereq == module:
                        cols[i+1].markdown("—", help="A module cannot be its own prerequisite")
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
            
            # Add vertical space after table
            st.markdown("<br>", unsafe_allow_html=True)
    
    else:  # Individual Module Editor (default)
        st.markdown("**Select prerequisites for each module individually**")
        st.caption("💡 Tip: Start with foundational modules (M1-M4) before configuring advanced ones")
        
        # Create expanders for each module
        for module in modules:
            with st.expander(f"{module} - {module_names[module]}"):
                # Get available prerequisites (all modules except itself)
                available_prereqs = [m for m in modules if m != module]
                
                # Format options with module names
                formatted_options = [f"{m} - {module_names[m]}" for m in available_prereqs]
                
                # Get current selections with proper formatting
                current_selections = st.session_state.prereqs.get(module, [])
                formatted_selections = [f"{m} - {module_names[m]}" for m in current_selections]
                
                # Multiselect with formatted display
                selected = st.multiselect(
                    f"Prerequisites for {module}:",
                    formatted_options,
                    default=formatted_selections,
                    key=f"multi_{module}"
                )
                
                # Convert back to module codes
                selected_codes = [opt.split(" - ")[0] for opt in selected]
                st.session_state.prereqs[module] = selected_codes
        
        # Add validation button
        if st.button("🔍 Validate Prerequisites"):
            # Check for missing modules in prerequisites
            invalid_prereqs = {}
            for module, prereqs in st.session_state.prereqs.items():
                for prereq in prereqs:
                    if prereq not in modules:
                        invalid_prereqs.setdefault(module, []).append(prereq)
            
            if invalid_prereqs:
                st.warning("⚠️ Found invalid prerequisites:")
                for module, invalids in invalid_prereqs.items():
                    st.write(f"- {module}: {', '.join(invalids)}")
                st.info("These prerequisites will be ignored during scheduling")
            else:
                st.success("✅ All prerequisites reference valid modules!")
            
            # Check for circular dependencies
            scheduler = CohortScheduler(
                modules=modules,
                prereqs=st.session_state.prereqs,
                cohort_starts={'C1': 1},  # Dummy cohort for validation
                config=SchedulerConfig(max_terms=10)
            )
            cycles = scheduler.detect_cycles()
            if cycles:
                st.error("❌ Found circular dependencies:")
                for cycle in cycles:
                    st.write(f"→ {' → '.join(cycle)} → {cycle[0]}")
                st.info("Fix these cycles before running the scheduler")
            else:
                st.success("✅ No circular dependencies detected!")

    # Display current configuration
    st.divider()
    st.subheader("Current Prerequisite Configuration")
    
    config_display = []
    for module in modules:
        prereq_list = st.session_state.prereqs.get(module, [])
        if prereq_list:
            prereq_names = [f"{p} ({module_names[p]})" for p in prereq_list if p in module_names]
            config_display.append(f"**{module} ({module_names[module]}):** {', '.join(prereq_names)}")
        else:
            config_display.append(f"**{module} ({module_names[module]}):** No prerequisites")
    
    st.markdown("\n\n".join(config_display))

with tab2:
    st.header("Run Scheduler")
    
    # Configuration options
    col1, col2 = st.columns(2)
    with col1:
        max_terms = st.number_input("Maximum Terms", min_value=10, max_value=200, value=50,
                                  help="Maximum number of terms to consider for scheduling")
    with col2:
        modules_per_term = st.number_input("Max Modules per Cohort per Term", min_value=1, max_value=5, value=1,
                                          help="How many modules a cohort can take in a single term")
    
    # Cohort configuration
    st.subheader("Cohort Start Terms")
    st.caption("Configure when each cohort begins their studies")
    
    cohort_starts = {}
    cohort_cols = st.columns(4)
    for i, cohort in enumerate([f'C{i+1}' for i in range(8)]):
        with cohort_cols[i % 4]:
            cohort_starts[cohort] = st.number_input(
                f"Start term for {cohort}",
                min_value=1,
                max_value=50,
                value=i*2 + 1,
                key=f"cohort_{cohort}"
            )
    
    if st.button("🚀 Run Scheduler", type="primary", use_container_width=True):
        with st.spinner("Running optimization..."):
            # Capture output
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            try:
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
                st.session_state.cohort_starts = cohort_starts
                
                output = sys.stdout.getvalue()
                sys.stdout = old_stdout
                
                if success:
                    st.success("✅ Scheduling completed successfully!")
                else:
                    st.warning("⚠️ Scheduling completed with incomplete cohorts")
                
                # Show summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Module Runs", summary['total_runs'])
                with col2:
                    st.metric("Schedule Length", f"{summary['max_term']} terms")
                with col3:
                    baseline_runs = 8 * 12  # 8 cohorts * 12 modules
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
        st.image("https://streamlit.io/images/hero.png", width=300, caption="Run scheduler to see results")
    else:
        scheduler = st.session_state.scheduler
        summary = st.session_state.summary
        cohort_starts = st.session_state.cohort_starts
        
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
            writer.writerow(['Cohort', 'Module_Code', 'Module_Name', 'Term_Taken', 'Start_Term'])
            for cohort in sorted(scheduler.cohort_progress.keys()):
                start_term = cohort_starts[cohort]
                for module, term in sorted(scheduler.cohort_progress[cohort].items(), key=lambda x: x[1]):
                    writer.writerow([cohort, module, module_names[module], term, start_term])
            return output.getvalue()
        
        def generate_metrics_csv():
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(['Metric', 'Baseline', 'Optimized', 'Improvement_Percent'])
            baseline_runs = 8 * 12  # 8 cohorts * 12 modules
            opt_runs = summary['total_runs']
            improvement = (baseline_runs - opt_runs) / baseline_runs * 100 if baseline_runs > 0 else 0
            writer.writerow(['Total_Module_Runs', baseline_runs, opt_runs, f"{improvement:.2f}"])
            writer.writerow(['Schedule_Length', 'N/A', summary['max_term'], 'N/A'])
            return output.getvalue()
        
        def generate_mapping_csv():
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(['Module_Code', 'Module_Name', 'Terms_Offered', 'Run_Count'])
            for module in sorted(module_names.keys()):
                terms = sorted(scheduler.module_runs[module])
                writer.writerow([module, module_names[module], ','.join(map(str, terms)), len(terms)])
            return output.getvalue()
        
        with col1:
            st.download_button(
                "📄 Term Schedule",
                generate_term_csv(),
                "term_by_term_schedule.csv",
                "text/csv",
                help="Schedule showing which modules run each term and which cohorts attend"
            )
        
        with col2:
            st.download_button(
                "📄 Cohort Progression",
                generate_cohort_csv(),
                "cohort_progression.csv",
                "text/csv",
                help="Complete progression of each cohort through all modules"
            )
        
        with col3:
            st.download_button(
                "📊 Metrics",
                generate_metrics_csv(),
                "metrics_comparison.csv",
                "text/csv",
                help="Comparison of optimized schedule vs baseline (one module run per cohort)"
            )
        
        with col4:
            st.download_button(
                "🔍 Module Mapping",
                generate_mapping_csv(),
                "module_term_mapping.csv",
                "text/csv",
                help="Mapping of which modules run in which terms"
            )
        
        # Display detailed results
        st.divider()
        
        # Term-by-term schedule
        st.subheader("📅 Term-by-Term Schedule")
        if not scheduler.schedule:
            st.info("No schedule generated. Check for errors in prerequisites or configuration.")
        else:
            terms = sorted(scheduler.schedule.keys())
            selected_term = st.select_slider("Select Term to View", options=terms, value=terms[0])
            
            if selected_term in scheduler.schedule and scheduler.schedule[selected_term]:
                st.markdown(f"### Term {selected_term}")
                for module, cohorts in sorted(scheduler.schedule[selected_term].items()):
                    st.markdown(f"**{module_names[module]}** ({module})")
                    st.markdown(f"Cohorts: {', '.join(sorted(cohorts))}")
                    st.markdown("---")
            else:
                st.info(f"No modules scheduled for Term {selected_term}")
        
        # Cohort progression
        st.subheader("👥 Cohort Progression")
        selected_cohort = st.selectbox("Select Cohort", options=sorted(scheduler.cohort_progress.keys()))
        
        if selected_cohort in scheduler.cohort_progress:
            cohort_modules = sorted(
                scheduler.cohort_progress[selected_cohort].items(),
                key=lambda x: x[1]
            )
            if cohort_modules:
                st.markdown(f"### Progression for Cohort {selected_cohort} (Started Term {cohort_starts[selected_cohort]})")
                for module, term in cohort_modules:
                    st.markdown(f"**Term {term}:** {module_names[module]} ({module})")
            else:
                st.info(f"Cohort {selected_cohort} has no scheduled modules")
        else:
            st.warning(f"Cohort {selected_cohort} not found in schedule")
        
        # Module-term mapping
        st.subheader("🗺️ Module-Term Mapping")
        selected_module = st.selectbox("Select Module", options=sorted(module_names.keys()))
        
        if selected_module in scheduler.module_runs:
            terms = sorted(scheduler.module_runs[selected_module])
            if terms:
                st.markdown(f"### {module_names[selected_module]} ({selected_module})")
                st.markdown(f"**Offered in terms:** {', '.join(map(str, terms))}")
                st.markdown(f"**Total runs:** {len(terms)}")
                
                # Visual timeline
                max_term = max(terms + [1])
                timeline = ["▢"] * (max_term + 1)
                for t in terms:
                    if t <= max_term:
                        timeline[t] = "✅"
                timeline_str = "".join(timeline[1:])
                st.markdown(f"**Term timeline:** `1`{''.join(timeline[1:])}`{max_term}`")
            else:
                st.info(f"Module {selected_module} is never scheduled")
        else:
            st.warning(f"Module {selected_module} not found in schedule")

# Footer
st.divider()
st.markdown("""
**💡 Tips for Better Schedules:**
- Start with foundational modules having no prerequisites
- Avoid circular dependencies (A requires B, B requires A)
- Increase 'Max Modules per Cohort per Term' for faster completion
- Adjust cohort start terms to balance resource usage
- Use presets as starting points for your configuration
""")

st.caption("Scheduler v2.1 • Handles 12 modules and 8 cohorts • Uses greedy optimization algorithm")
