"""
Streamlit Cohort Scheduler with Multiple Programmes & Shared Modules
Basel Barakat  (programme-aware refactor)

Model
-----
* One GLOBAL module pool (code -> name). New modules can be added freely.
* One GLOBAL prerequisite graph. A module's prerequisites are the same
  everywhere it appears. When a programme does not include a particular
  prerequisite, that prerequisite is simply ignored for that programme's
  cohorts (so a capstone can list every module and each programme only
  enforces the ones it actually teaches).
* A PROGRAMME is a named subset of the module pool.
* A COHORT belongs to one programme and has a start term.

Shared modules: the same module code appearing in two programmes is the
same module. The optimiser merges cohorts -- including cohorts from
different programmes -- into a single module run whenever they are all
eligible in the same term.
"""

import streamlit as st
from collections import defaultdict
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
import csv
import io
import re
import sys
import networkx as nx

TERM_DATES = [
    "31-Aug-26",    #T1 C1
    "26-Oct-26",    #T2 C2
    "04-Jan-27",    #T3
    "01-Mar-27",    #T4 C3
    "03-May-27",    #T5
    "28-Jun-27",    #T6 C4
    "30-Aug-27",    #T7 C5
    "30-Aug-27",    #T8
    "25-Oct-27",    #T9 C6
    "03-Jan-28",    #T10
    "28-Feb-28",    #T11 C7
    "01-May-28",    #T12
    "26-Jun-28",    #T13 C8
    "28-Aug-28",    #T14
    "23-Oct-28",    #T15 C9
    "01-Jan-29",    #T16
    "26-Feb-29"     #T17 C10
]

DEFAULT_COHORTS = [
    {"Cohort": "AI-1", "Programme": "MSc Artificial Intelligence & Machine Learning", "Start Term": 1},
    {"Cohort": "DA-1", "Programme": "MSc AI and Data Analytics", "Start Term": 3},
    {"Cohort": "CL-1", "Programme": "MSc Computational Linguistics", "Start Term": 3},
    {"Cohort": "AI-2", "Programme": "MSc Artificial Intelligence & Machine Learning", "Start Term": 3},
    {"Cohort": "DA-2", "Programme": "MSc AI and Data Analytics", "Start Term": 6},
    {"Cohort": "CL-2", "Programme": "MSc Computational Linguistics", "Start Term": 6},
]
# ========== DEFAULT DATA ==========

DEFAULT_MODULE_NAMES = {
    # Shared / AI core
    'M1': 'Research Methods',
    'M2': 'Programming & Algorithms',
    'M3': 'Data Programming',
    'M4': 'Artificial Intelligence',
    'M5': 'Machine Learning',
    'M6': 'Deep Learning',
    'M7': 'Natural Language Processing',
    'M8': 'Computer Vision',
    'M9': 'MLOps',
    'M10': 'Final Project 1',
    'M11': 'Final Project 2',
    'M12': 'Final Project 3',
    # Data Analytics
    'M13': 'Statistical Inference',
    'M14': 'Data Visualisation',
    'M15': 'Big Data',
    'M16': 'Data Mining',
    # Computational Linguistics
    'M17': 'Core Issues in Language and Linguistics',
    'M18': 'Corpus Linguistics',
    'M19': 'Interaction Science',
    'M20': 'Large Language Models',
    'M21': 'Final Project in Computational Linguistics 1',
    'M22': 'Final Project in Computational Linguistics 2',
    'M23': 'Final Project in Computational Linguistics 3',
    'M24': 'Final Project in Computational Linguistics 4',
}

# Global prerequisites (same everywhere). Capstones list every taught module;
# each programme only enforces the ones it teaches.
# Every taught (non-final-project) module. A capstone can list all of these as
# prerequisites; each programme only enforces the ones it actually teaches.
ALL_TAUGHT = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
              'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20']

# Modules taught on the Computational Linguistics programme (used for its
# final-project chain). Only enforced for cohorts that take them.
CL_TAUGHT = ['M17', 'M3', 'M18', 'M4', 'M5', 'M7', 'M19', 'M20']

DEFAULT_PREREQS = {
    'M1': [], 'M2': [], 'M3': [], 'M4': [],
    'M5': ['M2', 'M3'],
    'M6': ['M5'],
    'M7': ['M5'],
    'M8': ['M5'],
    'M9': ['M5'],
    # Data Analytics modules
    'M13': [],
    'M14': ['M3'],
    'M15': ['M2'],
    'M16': ['M3'],
    # Computational Linguistics modules
    'M17': [],
    'M18': ['M17'],
    'M19': [],
    'M20': ['M5', 'M7'],
    # Generic final-project chain (shared by AI & Data Analytics)
    'M10': list(ALL_TAUGHT),
    'M11': ['M10'],
    'M12': ['M11'],
    # Computational Linguistics final-project chain
    'M21': list(CL_TAUGHT),
    'M22': ['M21'],
    'M23': ['M22'],
    'M24': ['M23'],
}

DEFAULT_PROGRAMMES = {
    # Original programme (unchanged)
    "MSc Artificial Intelligence & Machine Learning": [
        'M1', 'M2', 'M3', 'M4', 'M5', 
        'M6','M7', 'M8', 'M9',
        'M10', 'M11', 'M12',
    ],
    # New: shared core (Research Methods, Programming, Data Programming, AI,
    # ML, MLOps) + DA-specific (Data Mining, NLP, Big Data, Data Visualisation)
    # + generic final projects. The DA-specific four are as you specified;
    # the core and final-project choices are assumptions to confirm.
    "MSc AI and Data Analytics": [
        'M1', 'M2', 'M3', 'M4', 'M5',          # shared core (assumed)
        'M16', 'M7', 'M15', 'M14',                   # DA-specific (given)
        'M10', 'M11', 'M12',                         # final projects (assumed)
    ],
    # New: exactly the 12 modules you listed
    "MSc Computational Linguistics": [
        'M17', 'M3', 'M18', 'M4', 'M5', 'M7', 'M19', 'M20',
        'M21', 'M22', 'M23', 'M24',
    ],
}



# Prerequisite presets (apply to whichever of these codes exist in the pool)
PRESETS = {
    "No Prerequisites (just project)": {
        'M1': [], 'M2': [], 'M3': [], 'M4': [], 'M5': [], 'M6': [],
        'M7': [], 'M8': [], 'M9': [], 'M13': [], 'M14': [], 'M15': [],
        'M10': list(ALL_TAUGHT), 'M11': ['M10'], 'M12': ['M11'],
    },
    "Logical": {
        'M1': [], 'M2': [], 'M3': [], 'M4': ['M2', 'M3'],
        'M5': ['M4'], 'M6': ['M5'], 'M7': ['M6'], 'M8': ['M6'], 'M9': ['M5'],
        'M13': [], 'M14': ['M3'], 'M15': ['M2'],
        'M10': list(ALL_TAUGHT), 'M11': ['M10'], 'M12': ['M11'],
    },
    "Sequential (M1..M9)": {
        'M1': [], 'M2': ['M1'], 'M3': ['M2'], 'M4': ['M3'], 'M5': ['M4'],
        'M6': ['M5'], 'M7': ['M6'], 'M8': ['M7'], 'M9': ['M8'],
        'M13': [], 'M14': ['M3'], 'M15': ['M2'],
        'M10': list(ALL_TAUGHT), 'M11': ['M10'], 'M12': ['M11'],
    },
}

def get_term_date(term: int) -> str:
    if 1 <= term <= len(TERM_DATES):
        return TERM_DATES[term - 1]
    return f"T{term}"


def mod_sort_key(code: str):
    """Natural sort so M2 < M10."""
    m = re.match(r'([A-Za-z]*)(\d+)$', str(code))
    if m:
        return (m.group(1), int(m.group(2)))
    return (str(code), 0)


def sorted_modules(codes) -> List[str]:
    return sorted(codes, key=mod_sort_key)


# ========== SCHEDULER ==========

@dataclass
class SchedulerConfig:
    max_terms: int = 100
    max_modules_per_cohort_per_term: int = 1
    verbose: bool = False


class CohortScheduler:
    """
    Programme-aware scheduler.

    Parameters
    ----------
    modules : list[str]
        The union of every module taught by any programme (used for run
        tracking and cycle detection).
    prereqs : dict[str, list[str]]
        Global prerequisite graph (same everywhere).
    programmes : dict[str, list[str]]
        Programme name -> list of module codes that programme teaches.
    cohort_config : dict[str, dict]
        Cohort name -> {'programme': <name>, 'start': <int>}.
    """

    def __init__(self, modules: List[str], prereqs: Dict[str, List[str]],
                 programmes: Dict[str, List[str]],
                 cohort_config: Dict[str, Dict[str, Any]],
                 config: Optional[SchedulerConfig] = None):
        self.config = config or SchedulerConfig()
        self.modules = list(modules)
        self.prereqs = prereqs
        self.programmes = programmes
        self.cohort_config = cohort_config
        self.cohorts = list(cohort_config.keys())

        self.cohort_starts = {c: cohort_config[c]['start'] for c in self.cohorts}
        self.cohort_programme = {c: cohort_config[c]['programme'] for c in self.cohorts}
        self.cohort_modules = {
            c: set(programmes.get(self.cohort_programme[c], []))
            for c in self.cohorts
        }
        self._reset_schedule()

    def _reset_schedule(self):
        self.schedule = defaultdict(lambda: defaultdict(list))
        self.cohort_progress = {c: {} for c in self.cohorts}
        self.module_runs = {m: [] for m in self.modules}
        self.cohort_last_active = {c: 0 for c in self.cohorts}
        self.cohort_module_count = {c: defaultdict(int) for c in self.cohorts}

    def can_take_module(self, cohort: str, module: str, term: int) -> bool:
        """Whether `cohort` may take `module` in `term`."""
        prog_modules = self.cohort_modules[cohort]

        # Module must belong to this cohort's programme
        if module not in prog_modules:
            return False
        if term < self.cohort_starts[cohort]:
            return False
        if self.cohort_module_count[cohort][term] >= self.config.max_modules_per_cohort_per_term:
            return False
        if module in self.cohort_progress[cohort]:
            return False

        # Prerequisites (global), but only those the programme actually teaches
        for prereq in self.prereqs.get(module, []):
            if prereq not in prog_modules:
                continue  # not part of this programme -> ignore
            if prereq not in self.cohort_progress[cohort]:
                return False
            if self.cohort_progress[cohort][prereq] >= term:
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

    def detect_cycles(self) -> List[List[str]]:
        """Cycle detection on the global prerequisite graph."""
        G = nx.DiGraph()
        module_set = set(self.modules)
        for module in self.modules:
            G.add_node(module)
            for prereq in self.prereqs.get(module, []):
                if prereq in module_set:
                    G.add_edge(prereq, module)
        try:
            return list(nx.simple_cycles(G))
        except nx.NetworkXError:
            return []

    def find_optimal_schedule(self) -> bool:
        cycles = self.detect_cycles()
        if cycles:
            error_msg = "❌ Found circular dependencies in prerequisites:\n"
            for cycle in cycles:
                error_msg += f"  → {' → '.join(cycle)} → {cycle[0]}\n"
            error_msg += "Please fix these cycles before scheduling."
            st.error(error_msg)
            return False

        self._reset_schedule()
        # Each cohort only needs the modules of its own programme
        cohort_needs = {c: set(self.cohort_modules[c]) for c in self.cohorts}

        for term in range(1, self.config.max_terms + 1):
            active_cohorts = [
                c for c in self.cohorts
                if term >= self.cohort_starts[c]
                and cohort_needs[c]
                and self.cohort_module_count[c][term] < self.config.max_modules_per_cohort_per_term
            ]
            if not active_cohorts:
                continue

            scheduled_this_term: Set[str] = set()
            term_scheduled = False

            while True:
                unscheduled = [c for c in active_cohorts if c not in scheduled_this_term]
                if not unscheduled:
                    break

                # Candidate modules = anything still needed by any unscheduled
                # cohort. Scoring by number of eligible cohorts maximises
                # sharing, including across different programmes.
                candidate_modules: Set[str] = set()
                for c in unscheduled:
                    candidate_modules |= cohort_needs[c]

                module_scores = []
                for module in sorted_modules(candidate_modules):
                    eligible = [c for c in unscheduled if self.can_take_module(c, module, term)]
                    if eligible:
                        module_scores.append((len(eligible), module, eligible))

                if not module_scores:
                    break

                module_scores.sort(reverse=True, key=lambda x: x[0])
                _, best_module, eligible_cohorts = module_scores[0]

                self.schedule_module_run(best_module, term, eligible_cohorts)
                term_scheduled = True

                for cohort in eligible_cohorts:
                    cohort_needs[cohort].discard(best_module)
                    scheduled_this_term.add(cohort)

            if not term_scheduled:
                # Nothing schedulable this term; only stop if no cohort can
                # ever progress (all remaining cohorts are blocked/finished).
                if all(not cohort_needs[c] for c in self.cohorts):
                    break

        all_done = all(
            len(self.cohort_progress[c]) == len(self.cohort_modules[c])
            for c in self.cohorts
        )

        if not all_done:
            incomplete = [c for c in self.cohorts
                          if len(self.cohort_progress[c]) < len(self.cohort_modules[c])]
            missing_modules = {}
            for cohort in incomplete:
                missing = self.cohort_modules[cohort] - set(self.cohort_progress[cohort].keys())
                missing_modules[cohort] = missing

            warning_msg = "⚠️ Schedule incomplete. These cohorts couldn't finish their programme:\n"
            for cohort, mods in missing_modules.items():
                prog = self.cohort_programme[cohort]
                warning_msg += f"- {cohort} ({prog}): missing {', '.join(sorted_modules(mods))}\n"
            warning_msg += "\nPossible causes:\n"
            warning_msg += "- Prerequisites create impossible sequences\n"
            warning_msg += "- Not enough terms allocated\n"
            warning_msg += "- Too few modules allowed per term"
            st.warning(warning_msg)

        return all_done

    def get_schedule_summary(self) -> Dict[str, Any]:
        total_runs = sum(len(runs) for runs in self.module_runs.values())
        max_term = max(self.cohort_last_active.values()) if self.cohort_last_active else 0
        # Baseline = every cohort runs every one of its programme's modules alone
        baseline_runs = sum(len(self.cohort_modules[c]) for c in self.cohorts)
        return {
            'total_runs': total_runs,
            'max_term': max_term,
            'baseline_runs': baseline_runs,
            'module_runs': {m: len(runs) for m, runs in self.module_runs.items()},
            'cohort_progress': self.cohort_progress,
            'schedule': self.schedule,
        }




# ========== SESSION STATE ==========

def init_state():
    if 'module_names' not in st.session_state:
        st.session_state.module_names = dict(DEFAULT_MODULE_NAMES)
    if 'prereqs' not in st.session_state:
        st.session_state.prereqs = {k: list(v) for k, v in DEFAULT_PREREQS.items()}
    if 'programmes' not in st.session_state:
        st.session_state.programmes = {k: list(v) for k, v in DEFAULT_PROGRAMMES.items()}
    if 'cohorts' not in st.session_state:
        st.session_state.cohorts = [dict(c) for c in DEFAULT_COHORTS]
    # Keep prereqs in sync with the module pool
    for m in st.session_state.module_names:
        st.session_state.prereqs.setdefault(m, [])


st.set_page_config(page_title="Cohort Scheduler", page_icon="📚", layout="wide")
init_state()

st.title("📚 Cohort Scheduler — Multiple Programmes")
st.markdown("Define a shared module pool, build programmes from it, assign cohorts, and optimise the schedule. "
            "Cohorts from different programmes share a module run whenever they take the same module in the same term.")

# Convenient references
module_names: Dict[str, str] = st.session_state.module_names


def pool_modules() -> List[str]:
    return sorted_modules(st.session_state.module_names.keys())


def used_modules() -> List[str]:
    """Union of all modules referenced by any programme."""
    u: Set[str] = set()
    for mods in st.session_state.programmes.values():
        u |= set(mods)
    return sorted_modules(u)


def label(code: str) -> str:
    return f"{module_names.get(code, code)} ({code})"


# ========== SIDEBAR: prerequisite presets ==========
st.sidebar.header("⚙️ Prerequisite Presets")
preset = st.sidebar.selectbox("Load preset:", ["Custom"] + list(PRESETS.keys()),
                              key="preset_selector")
if st.sidebar.button("✅ Load Preset"):
    if preset in PRESETS:
        for m in pool_modules():
            st.session_state.prereqs[m] = list(PRESETS[preset].get(m, []))
        st.sidebar.success(f"Loaded '{preset}' prerequisites.")
        st.rerun()
    else:
        st.sidebar.info("Custom configuration preserved.")

st.sidebar.divider()
st.sidebar.caption("Presets only set prerequisites for module codes that exist "
                   "in the pool. Programmes and cohorts are unaffected.")


# ========== TABS ==========
tab_mod, tab_prog, tab_run, tab_results = st.tabs(
    ["📦 Modules & Prerequisites", "🎓 Programmes", "👥 Cohorts & Run", "📊 Results"]
)

# ---------- Tab: Modules & Prerequisites ----------
with tab_mod:
    st.header("Module Pool")
    st.markdown("Add, rename or remove modules in the shared pool. Every programme draws from this pool.")

    pool_df_data = [{"Code": c, "Name": module_names[c]} for c in pool_modules()]
    edited = st.data_editor(
        pool_df_data,
        num_rows="dynamic",
        use_container_width=True,
        key="module_pool_editor",
        column_config={
            "Code": st.column_config.TextColumn("Code", help="Unique short code, e.g. M16", required=True),
            "Name": st.column_config.TextColumn("Name", required=True),
        },
    )

    if st.button("💾 Save Module Pool"):
        new_names: Dict[str, str] = {}
        problems = []
        for row in edited:
            code = (row.get("Code") or "").strip()
            name = (row.get("Name") or "").strip()
            if not code:
                continue
            if code in new_names:
                problems.append(f"Duplicate code '{code}'")
                continue
            new_names[code] = name or code
        if problems:
            st.error("Couldn't save: " + "; ".join(problems))
        else:
            st.session_state.module_names = new_names
            # Prune prereqs/programmes that reference removed modules
            valid = set(new_names)
            for m in list(st.session_state.prereqs.keys()):
                if m not in valid:
                    del st.session_state.prereqs[m]
            for m in valid:
                st.session_state.prereqs.setdefault(m, [])
                st.session_state.prereqs[m] = [p for p in st.session_state.prereqs[m] if p in valid]
            for prog in st.session_state.programmes:
                st.session_state.programmes[prog] = [
                    c for c in st.session_state.programmes[prog] if c in valid
                ]
            st.success("Module pool saved.")
            st.rerun()

    st.divider()
    st.header("Global Prerequisites")
    st.caption("A module's prerequisites are the same in every programme. "
               "If a programme doesn't teach a listed prerequisite, it's ignored for that programme.")

    all_mods = pool_modules()
    for module in all_mods:
        with st.expander(f"{module} — {module_names[module]}"):
            options = [m for m in all_mods if m != module]
            formatted_options = [label(m) for m in options]
            current = [p for p in st.session_state.prereqs.get(module, []) if p in options]
            formatted_current = [label(m) for m in current]
            selected = st.multiselect(
                f"Prerequisites for {module}:",
                formatted_options,
                default=formatted_current,
                key=f"multi_{module}",
            )
            codes = []
            for opt in selected:
                m = re.search(r'\(([^()]+)\)\s*$', opt)
                if m:
                    codes.append(m.group(1))
            st.session_state.prereqs[module] = codes

    if st.button("🔍 Validate Prerequisites"):
        scheduler = CohortScheduler(
            modules=all_mods,
            prereqs=st.session_state.prereqs,
            programmes={"_all": all_mods},
            cohort_config={'C1': {'programme': '_all', 'start': 1}},
            config=SchedulerConfig(max_terms=10),
        )
        cycles = scheduler.detect_cycles()
        if cycles:
            st.error("❌ Circular dependencies detected:")
            for cycle in cycles:
                st.write(f"→ {' → '.join(cycle)} → {cycle[0]}")
        else:
            st.success("✅ No circular dependencies detected.")

# ---------- Tab: Programmes ----------
with tab_prog:
    st.header("Programmes")
    st.markdown("Each programme is a subset of the module pool. A module shared by two programmes "
                "is the **same** module, and the scheduler will merge cohorts into one run when possible.")

    all_mods = pool_modules()

    # Add a new programme
    with st.form("add_programme", clear_on_submit=True):
        new_prog = st.text_input("New programme name")
        add = st.form_submit_button("➕ Add Programme")
        if add:
            name = new_prog.strip()
            if not name:
                st.warning("Give the programme a name.")
            elif name in st.session_state.programmes:
                st.warning("A programme with that name already exists.")
            else:
                st.session_state.programmes[name] = []
                st.success(f"Added '{name}'.")
                st.rerun()

    st.divider()

    if not st.session_state.programmes:
        st.info("No programmes yet — add one above.")

    for prog in list(st.session_state.programmes.keys()):
        with st.expander(f"🎓 {prog}  ·  {len(st.session_state.programmes[prog])} modules", expanded=True):
            formatted_options = [label(m) for m in all_mods]
            current = [m for m in st.session_state.programmes[prog] if m in all_mods]
            formatted_current = [label(m) for m in current]
            selected = st.multiselect(
                "Modules in this programme:",
                formatted_options,
                default=formatted_current,
                key=f"prog_{prog}",
            )
            codes = []
            for opt in selected:
                m = re.search(r'\(([^()]+)\)\s*$', opt)
                if m:
                    codes.append(m.group(1))
            st.session_state.programmes[prog] = sorted_modules(codes)

            if st.button(f"🗑️ Delete '{prog}'", key=f"del_{prog}"):
                del st.session_state.programmes[prog]
                # Drop cohorts pointing at the deleted programme
                st.session_state.cohorts = [
                    c for c in st.session_state.cohorts if c.get("Programme") != prog
                ]
                st.rerun()

    # Shared-module overview
    st.divider()
    st.subheader("Shared modules across programmes")
    progs = st.session_state.programmes
    if len(progs) >= 2:
        usage = defaultdict(list)
        for p, mods in progs.items():
            for m in mods:
                usage[m].append(p)
        shared = {m: ps for m, ps in usage.items() if len(ps) >= 2}
        if shared:
            for m in sorted_modules(shared.keys()):
                st.write(f"- **{label(m)}** → shared by: {', '.join(shared[m])}")
        else:
            st.caption("No modules are currently shared between programmes.")
    else:
        st.caption("Add at least two programmes to see shared modules.")

# ---------- Tab: Cohorts & Run ----------
with tab_run:
    st.header("Cohorts")
    st.caption("Assign each cohort to a programme and a start term.")

    prog_options = list(st.session_state.programmes.keys())
    if not prog_options:
        st.warning("Define at least one programme first (Programmes tab).")
    else:
        # Make sure existing cohort rows point at a valid programme
        for c in st.session_state.cohorts:
            if c.get("Programme") not in prog_options:
                c["Programme"] = prog_options[0]

        edited_cohorts = st.data_editor(
            st.session_state.cohorts,
            num_rows="dynamic",
            use_container_width=True,
            key="cohort_editor",
            column_config={
                "Cohort": st.column_config.TextColumn("Cohort", required=True),
                "Programme": st.column_config.SelectboxColumn(
                    "Programme", options=prog_options, required=True),
                "Start Term": st.column_config.NumberColumn(
                    "Start Term", min_value=1, max_value=200, step=1, default=1),
            },
        )
        st.session_state.cohorts = edited_cohorts

    st.divider()
    st.header("Run Scheduler")
    col1, col2 = st.columns(2)
    with col1:
        max_terms = st.number_input("Maximum Terms", min_value=10, max_value=200, value=60,
                                    help="Maximum number of terms to consider")
    with col2:
        modules_per_term = st.number_input("Max Modules per Cohort per Term",
                                           min_value=1, max_value=5, value=1)

    if st.button("🚀 Run Scheduler", type="primary", use_container_width=True):
        # Build cohort_config from the editor
        cohort_config = {}
        valid = True
        seen = set()
        for row in st.session_state.cohorts:
            name = (row.get("Cohort") or "").strip()
            prog = row.get("Programme")
            start = int(row.get("Start Term") or 1)
            if not name:
                continue
            if name in seen:
                st.error(f"Duplicate cohort name '{name}'.")
                valid = False
                break
            if prog not in st.session_state.programmes:
                st.error(f"Cohort '{name}' has no valid programme.")
                valid = False
                break
            seen.add(name)
            cohort_config[name] = {'programme': prog, 'start': start}

        if valid and not cohort_config:
            st.error("Add at least one cohort.")
            valid = False

        if valid:
            with st.spinner("Running optimisation..."):
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                try:
                    config = SchedulerConfig(
                        max_terms=max_terms,
                        max_modules_per_cohort_per_term=modules_per_term,
                    )
                    scheduler = CohortScheduler(
                        modules=used_modules(),
                        prereqs=st.session_state.prereqs,
                        programmes=st.session_state.programmes,
                        cohort_config=cohort_config,
                        config=config,
                    )
                    success = scheduler.find_optimal_schedule()
                    summary = scheduler.get_schedule_summary()

                    st.session_state.scheduler = scheduler
                    st.session_state.summary = summary
                    st.session_state.success = success
                    sys.stdout = old_stdout

                    if success:
                        st.success("✅ Scheduling completed successfully!")
                    else:
                        st.warning("⚠️ Scheduling completed with incomplete cohorts.")

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Total Module Runs", summary['total_runs'])
                    with c2:
                        st.metric("Schedule Length", f"{summary['max_term']} terms")
                    with c3:
                        base = summary['baseline_runs']
                        improvement = (base - summary['total_runs']) / base * 100 if base else 0
                        st.metric("Improvement vs Baseline", f"{improvement:.1f}%")

                    st.info("✨ Go to the 'Results' tab for the detailed schedule and downloads.")
                except Exception as e:
                    sys.stdout = old_stdout
                    st.error(f"❌ Error running scheduler: {e}")
                    st.exception(e)

# ---------- Tab: Results ----------
with tab_results:
    st.header("Results & Downloads")

    if 'scheduler' not in st.session_state:
        st.info("👈 Run the scheduler first in the 'Cohorts & Run' tab.")
    else:
        scheduler: CohortScheduler = st.session_state.scheduler
        summary = st.session_state.summary
        max_term = summary['max_term']

        def cohort_label(c: str) -> str:
            return f"{c} · {scheduler.cohort_programme.get(c, '?')}"

        # ----- Downloads -----
        st.subheader("📥 Download CSV Files")

        def generate_term_csv():
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(['Term', 'Date', 'Module_Code', 'Module_Name', 'Cohorts', 'Programmes'])
            for term in sorted(scheduler.schedule.keys()):
                for module in sorted_modules(scheduler.schedule[term].keys()):
                    cohorts = sorted(scheduler.schedule[term][module])
                    progs = sorted({scheduler.cohort_programme[c] for c in cohorts})
                    writer.writerow([term, get_term_date(term), module,
                                     module_names.get(module, module),
                                     ','.join(cohorts), '; '.join(progs)])
            return output.getvalue()

        def generate_cohort_csv():
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(['Cohort', 'Programme', 'Module_Code', 'Module_Name', 'Term_Taken', 'Start_Term'])
            for cohort in sorted(scheduler.cohort_progress.keys()):
                start = scheduler.cohort_starts[cohort]
                prog = scheduler.cohort_programme[cohort]
                for module, term in sorted(scheduler.cohort_progress[cohort].items(), key=lambda x: x[1]):
                    writer.writerow([cohort, prog, module, module_names.get(module, module), term, start])
            return output.getvalue()

        def generate_metrics_csv():
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(['Metric', 'Baseline', 'Optimised', 'Improvement_Percent'])
            base = summary['baseline_runs']
            opt = summary['total_runs']
            imp = (base - opt) / base * 100 if base else 0
            writer.writerow(['Total_Module_Runs', base, opt, f"{imp:.2f}"])
            writer.writerow(['Schedule_Length', 'N/A', summary['max_term'], 'N/A'])
            return output.getvalue()

        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("📄 Term Schedule", generate_term_csv(),
                               "term_schedule.csv", "text/csv")
        with c2:
            st.download_button("📄 Cohort Progression", generate_cohort_csv(),
                               "cohort_progression.csv", "text/csv")
        with c3:
            st.download_button("📊 Metrics", generate_metrics_csv(),
                               "metrics_comparison.csv", "text/csv")

        # ----- Term x Cohort matrix -----
        st.divider()
        st.subheader("📊 Term × Cohort Matrix")
        if not scheduler.schedule:
            st.info("No schedule generated.")
        else:
            import pandas as pd
            header = ["Term", "Date"] + [cohort_label(c) for c in sorted(scheduler.cohorts)]
            rows = []
            for term in range(1, max_term + 1):
                row = [f"T{term}", get_term_date(term)]
                for cohort in sorted(scheduler.cohorts):
                    mods = []
                    if term in scheduler.schedule:
                        for module, cohorts in scheduler.schedule[term].items():
                            if cohort in cohorts:
                                mods.append(module_names.get(module, module))
                    row.append("\n".join(mods))
                rows.append(row)

            df = pd.DataFrame(rows, columns=header)
            styled = df.style.set_properties(**{
                'text-align': 'left', 'white-space': 'pre-wrap', 'vertical-align': 'top'
            }).set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#f0f2f6'), ('font-weight', 'bold')]},
                {'selector': 'td', 'props': [('border', '1px solid #e0e0e0'), ('padding', '8px')]},
            ])
            st.dataframe(styled, height=600, use_container_width=True)

            def generate_matrix_csv():
                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow(header)
                for term in range(1, max_term + 1):
                    row = [f"T{term}", get_term_date(term)]
                    for cohort in sorted(scheduler.cohorts):
                        mods = []
                        if term in scheduler.schedule:
                            for module, cohorts in scheduler.schedule[term].items():
                                if cohort in cohorts:
                                    mods.append(module_names.get(module, module))
                        row.append("; ".join(mods))
                    writer.writerow(row)
                return output.getvalue()

            st.download_button("📥 Download Matrix (CSV)", generate_matrix_csv(),
                               "term_cohort_matrix.csv", "text/csv")

        # ----- Shared runs highlight -----
        st.divider()
        st.subheader("🤝 Cross-programme shared runs")
        shared_runs = []
        for term in sorted(scheduler.schedule.keys()):
            for module in sorted_modules(scheduler.schedule[term].keys()):
                cohorts = scheduler.schedule[term][module]
                progs = {scheduler.cohort_programme[c] for c in cohorts}
                if len(progs) >= 2:
                    shared_runs.append(
                        f"T{term} · {module_names.get(module, module)}: "
                        f"{', '.join(sorted(cohorts))}  ({', '.join(sorted(progs))})"
                    )
        if shared_runs:
            st.markdown("These runs combine cohorts from **different programmes** into one delivery:")
            for line in shared_runs:
                st.write(f"- {line}")
        else:
            st.caption("No runs combined cohorts from different programmes "
                       "(they may still share runs within the same programme).")

        # ----- Per-cohort progression -----
        st.divider()
        st.subheader("👥 Cohort Progression")
        lines = []
        for cohort in sorted(scheduler.cohort_progress.keys()):
            prog = scheduler.cohort_programme[cohort]
            ordered = sorted(scheduler.cohort_progress[cohort].items(), key=lambda x: x[1])
            seq = "  ".join(f"{module_names.get(m, m)}(T{t})" for m, t in ordered)
            lines.append(f"{cohort} [{prog}]:  {seq}")
        st.text("\n".join(lines) if lines else "No progression data.")


# ----- Footer -----
st.divider()
st.markdown("""
**💡 Tips:**
- Add modules in the pool first, then build programmes from them.
- Shared modules are simply the same code appearing in more than one programme.
- A capstone can list every taught module as a prerequisite; each programme only enforces the ones it teaches.
- Stagger cohort start terms to encourage cross-programme sharing of foundational modules.
""")
st.caption("Scheduler v3.0 • Multiple programmes • Shared modules • Greedy optimisation")
# """
# Streamlit Cohort Scheduler with Customisable Prerequisites
# Basel Barakat
# """

# import streamlit as st
# from collections import defaultdict
# from typing import Dict, List, Optional, Any, Set
# from dataclasses import dataclass
# import csv
# import io
# import sys
# import networkx as nx  # Added for dependency cycle detection

# TERM_DATES = [
#     "08-Jun-26", "31-Aug-26", "26-Oct-26",
#     "04-Jan-27", "01-Mar-27", "03-May-27",
#     "08-Jun-27", "28-Jun-27", "30-Aug-27",
#     "31-Aug-27", "25-Oct-27", "26-Oct-27",
#     "03-Jan-28", "04-Jan-28", "28-Feb-28",
#     "01-Mar-28", "01-May-28", "03-May-28",
#     "26-Jun-28", "28-Jun-28", "28-Aug-28",
#     "30-Aug-28", "25-Oct-28", "03-Jan-29",
#     "28-Feb-29", "01-Mar-29"
# ]

# # Helper function to get term date
# def get_term_date(term: int) -> str:
#     if 1 <= term <= len(TERM_DATES):
#         return TERM_DATES[term-1]
#     return f"T{term}"


# # ========== SCHEDULER CLASSES ==========

# @dataclass
# class SchedulerConfig:
#     max_terms: int = 100
#     max_modules_per_cohort_per_term: int = 1
#     verbose: bool = False

# class CohortScheduler:
#     def __init__(self, modules: List[str], prereqs: Dict[str, List[str]], 
#                  cohort_starts: Dict[str, int], config: Optional[SchedulerConfig] = None):
#         self.config = config or SchedulerConfig()
#         self.modules = modules
#         self.prereqs = prereqs
#         self.cohort_starts = cohort_starts
#         self.cohorts = list(cohort_starts.keys())
#         self._reset_schedule()
    
#     def _reset_schedule(self):
#         self.schedule = defaultdict(lambda: defaultdict(list))
#         self.cohort_progress = {c: {} for c in self.cohorts}
#         self.module_runs = {m: [] for m in self.modules}
#         self.cohort_last_active = {c: 0 for c in self.cohorts}
#         self.cohort_module_count = {c: defaultdict(int) for c in self.cohorts}
    
#     def can_take_module(self, cohort: str, module: str, term: int) -> bool:
#         """Check if a cohort can take a module in a given term"""
#         # Basic checks
#         if term < self.cohort_starts[cohort]:
#             return False
#         if self.cohort_module_count[cohort][term] >= self.config.max_modules_per_cohort_per_term:
#             return False
#         if module in self.cohort_progress[cohort]:
#             return False
        
#         # Check prerequisites
#         for prereq in self.prereqs.get(module, []):
#             # Skip invalid prerequisites not in our module list
#             if prereq not in self.modules:
#                 continue
#             if prereq not in self.cohort_progress[cohort]:
#                 return False
#             if self.cohort_progress[cohort][prereq] >= term:
#                 return False
#         return True
    
#     def schedule_module_run(self, module: str, term: int, cohorts: List[str]):
#         """Schedule a module run for multiple cohorts"""
#         self.schedule[term][module].extend(cohorts)
#         for cohort in cohorts:
#             self.cohort_progress[cohort][module] = term
#             self.cohort_last_active[cohort] = max(self.cohort_last_active[cohort], term)
#             self.cohort_module_count[cohort][term] += 1
#         if term not in self.module_runs[module]:
#             self.module_runs[module].append(term)
    
#     def detect_cycles(self) -> List[List[str]]:
#         """Detect cycles in prerequisites using networkx"""
#         G = nx.DiGraph()
#         for module in self.modules:
#             G.add_node(module)
#             for prereq in self.prereqs.get(module, []):
#                 if prereq in self.modules:  # Only consider valid modules
#                     G.add_edge(prereq, module)
        
#         try:
#             cycles = list(nx.simple_cycles(G))
#             return cycles
#         except nx.NetworkXError:
#             return []
    
#     def find_optimal_schedule(self) -> bool:
#         """Find an optimal schedule for all cohorts"""
#         # First check for dependency cycles
#         cycles = self.detect_cycles()
#         if cycles:
#             error_msg = "❌ Found circular dependencies in prerequisites:\n"
#             for cycle in cycles:
#                 error_msg += f"  → {' → '.join(cycle)} → {cycle[0]}\n"
#             error_msg += "Please fix these cycles before scheduling."
#             st.error(error_msg)
#             return False
        
#         self._reset_schedule()
#         cohort_needs = {c: set(self.modules) for c in self.cohorts}
        
#         # Track unscheduled modules to detect impossible schedules
#         unscheduled_modules = set(self.modules)
        
#         for term in range(1, self.config.max_terms + 1):
#             active_cohorts = [
#                 c for c in self.cohorts
#                 if term >= self.cohort_starts[c] 
#                 and cohort_needs[c]
#                 and self.cohort_module_count[c][term] < self.config.max_modules_per_cohort_per_term
#             ]
            
#             if not active_cohorts:
#                 continue
            
#             scheduled_this_term = set()
#             term_scheduled = False
            
#             # Try to schedule modules for this term
#             while active_cohorts:
#                 unscheduled = [c for c in active_cohorts if c not in scheduled_this_term]
#                 if not unscheduled:
#                     break
                
#                 # Find best module to schedule
#                 module_scores = []
#                 for module in sorted(unscheduled_modules):  # Sort for deterministic behavior
#                     if module not in cohort_needs[unscheduled[0]]:
#                         continue
                    
#                     eligible = [c for c in unscheduled if self.can_take_module(c, module, term)]
                    
#                     if eligible:
#                         score = len(eligible)
#                         module_scores.append((score, module, eligible))
                
#                 if not module_scores:
#                     break
                
#                 # Select module with most eligible cohorts
#                 module_scores.sort(reverse=True, key=lambda x: x[0])
#                 _, best_module, eligible_cohorts = module_scores[0]
                
#                 self.schedule_module_run(best_module, term, eligible_cohorts)
#                 term_scheduled = True
                
#                 # Update cohort needs
#                 for cohort in eligible_cohorts:
#                     cohort_needs[cohort].discard(best_module)
#                     scheduled_this_term.add(cohort)
                
#                 # Remove module from unscheduled if all cohorts can take it eventually
#                 if all(best_module not in cohort_needs[c] for c in self.cohorts):
#                     unscheduled_modules.discard(best_module)
            
#             # Early termination if nothing was scheduled this term
#             if not term_scheduled:
#                 break
        
#         # Check completion status
#         all_modules_scheduled = all(len(self.cohort_progress[c]) == len(self.modules) for c in self.cohorts)
        
#         # Handle incomplete schedules
#         if not all_modules_scheduled:
#             incomplete_cohorts = [c for c in self.cohorts if len(self.cohort_progress[c]) < len(self.modules)]
#             missing_modules = {}
#             for cohort in incomplete_cohorts:
#                 missing = set(self.modules) - set(self.cohort_progress[cohort].keys())
#                 missing_modules[cohort] = missing
            
#             warning_msg = "⚠️ Schedule incomplete. The following cohorts couldn't finish all modules:\n"
#             for cohort, modules in missing_modules.items():
#                 warning_msg += f"- Cohort {cohort}: Missing {', '.join(sorted(modules))}\n"
#             warning_msg += "\nPossible causes:\n"
#             warning_msg += "- Prerequisites create impossible sequences\n"
#             warning_msg += "- Not enough terms allocated\n"
#             warning_msg += "- Too few modules allowed per term"
#             st.warning(warning_msg)
        
#         return all_modules_scheduled
    
#     def get_schedule_summary(self) -> Dict[str, Any]:
#         total_runs = sum(len(runs) for runs in self.module_runs.values())
#         max_term = max(self.cohort_last_active.values()) if self.cohort_last_active else 0
        
#         return {
#             'total_runs': total_runs,
#             'max_term': max_term,
#             'module_runs': {m: len(runs) for m, runs in self.module_runs.items()},
#             'cohort_progress': self.cohort_progress,
#             'schedule': self.schedule
#         }

# # ========== PRESET DEFINITIONS ==========

# PRESETS = {
#     "No Prerequisites (just project)": {
#         'M1': [], 'M2': [], 'M3': [], 'M4': [], 'M5': [], 'M6': [],
#         'M7': [], 'M8': [], 'M9': [],
#         'M10': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'],
#         'M11': ['M10'],
#         'M12': ['M11']
#     },
#     "Phase-Based": {
#         'M1': [], 'M2': [], 'M3': [], 'M4': [],#phase one
#         'M5': ['M1','M2', 'M3', 'M4'],
#         'M6': ['M1','M2', 'M3', 'M4'],
#         'M7': ['M1','M2', 'M3', 'M4'],
#         'M8': ['M1','M2', 'M3', 'M4'],
#         'M9': ['M5', 'M6', 'M7', 'M8'],
#         'M10': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'],
#         'M11': ['M10'],
#         'M12': ['M11']
#     },
#     "Logical": {
#         'M1': [], 'M2': [], 'M3': [],
#         'M4': ['M2', 'M3'],
#         'M5': ['M4'],
#         'M6': ['M5'],
#         'M7': ['M6'],
#         'M8': ['M6'],
#         'M9': ['M5'],
#         'M10': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'],
#         'M11': ['M10'],
#         'M12': ['M11']
#     },
    
#         "Logical (no project)": {
#         'M1': [], 'M2': [], 'M3': [],
#         'M4': ['M2', 'M3'],
#         'M5': ['M4'],
#         'M6': ['M5'],
#         'M7': ['M6'],
#         'M8': ['M6'],
#         'M9': ['M5'],
#         'M10': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6'],
#         'M11': ['M10'],
#         'M12': ['M11']
#     },
#     "Sequental": {
#         'M1': [], 'M2': ['M1'], 'M3': ['M2'],
#         'M4': ['M3'],
#         'M5': ['M4'],
#         'M6': ['M5'],
#         'M7': ['M6'],
#         'M8': ['M7'],
#         'M9': ['M8'],
#         'M10': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'],
#         'M11': ['M10'],
#         'M12': ['M11']
#     }
# }

# # ========== STREAMLIT APP ==========

# st.set_page_config(page_title="Cohort Scheduler", page_icon="📚", layout="wide")

# st.title("📚 Cohort Scheduler with Custom Prerequisites")
# st.markdown("Configure module prerequisites and run the scheduling optimisation.")

# # Module names mapping
# module_names = {
#     'M1': 'Research Methods',
#     'M2': 'Programming & Algorithms',
#     'M3': 'Data Programming',
#     'M4': 'Artificial Intelligence',
#     'M5': 'Machine Learning',
#     'M6': 'Deep Learning',
#     'M7': 'NLP',
#     'M8': 'Computer Vision',
#     'M9': 'MLOps',
#     'M10': 'Final Project 1',
#     'M11': 'Final Project 2',
#     'M12': 'Final Project 3'
# }

# modules = [f'M{i}' for i in range(1, 13)]

# # Initialise session state with proper defaults
# if 'prereqs' not in st.session_state:
#     st.session_state.prereqs = PRESETS["No Prerequisites (just project)"].copy()
#     # Ensure all modules exist in prerequisites
#     for module in modules:
#         if module not in st.session_state.prereqs:
#             st.session_state.prereqs[module] = []

# # Sidebar for preset configurations
# st.sidebar.header("⚙️ Configuration Presets")

# preset = st.sidebar.selectbox(
#     "Load Preset:",
#     ["Custom", "No Prerequisites (just project)", "Phase-Based", "Sequental","Logical","Logical (no project)"],
#     key="preset_selector"
# )

# # Reset to initial state button
# if st.sidebar.button("↩️ Reset to Initial"):
#     st.session_state.prereqs = PRESETS["No Prerequisites (just project)"].copy()
#     # Ensure all modules exist after reset
#     for module in modules:
#         if module not in st.session_state.prereqs:
#             st.session_state.prereqs[module] = []
#     st.success("✅ Reset to initial configuration!")
#     st.rerun()

# # Load preset button - FIXED THE LOGIC HERE
# if st.sidebar.button("✅ Load Preset"):
#     if preset != "Custom" and preset in PRESETS:
#         # Clear existing prerequisites first
#         st.session_state.prereqs = {}
#         # Load the selected preset
#         st.session_state.prereqs = PRESETS[preset].copy()
#         # Ensure all modules exist after loading preset
#         for module in modules:
#             if module not in st.session_state.prereqs:
#                 st.session_state.prereqs[module] = []
#         st.sidebar.success(f"✅ {preset} preset loaded!")
#         # Force a rerun to update the UI immediately
#         st.rerun()
#     else:
#         st.sidebar.info("Custom configuration preserved or preset not found")

# # Main content tabs
# tab1, tab2, tab3 = st.tabs(["📝 Edit Prerequisites", "▶️ Run Scheduler", "📊 Results"])

# with tab1:
#     st.header("Configure Module Prerequisites")
#     st.markdown("Select which modules are required before each module can be taken.")
    
#     # Edit mode selector
#     edit_mode = st.radio(
#         "Edit Mode:",
#         ["Individual Module Editor (Recommended)", "Table View"],
#         horizontal=True,
#         key="edit_mode"
#     )
    
#     if edit_mode == "Table View":
#         st.markdown("**Check the boxes to set prerequisites. Rows = modules, Columns = prerequisites**")
#         st.caption("💡 Tip: Use the individual editor for complex configurations")
        
#         # Create scrollable container for the large table
#         with st.container():
#             # Column headers
#             header_cols = st.columns([2.5] + [0.8] * len(modules))
#             header_cols[0].markdown("**Module**")
#             for i, mod in enumerate(modules):
#                 header_cols[i+1].markdown(f"**{mod}**", help=module_names[mod])
            
#             # Create rows for each module with scrollable container
#             for module in modules:
#                 cols = st.columns([2.5] + [0.8] * len(modules))
#                 cols[0].markdown(f"**{module}** - {module_names[module]}")
                
#                 # Checkboxes for each potential prerequisite
#                 for i, potential_prereq in enumerate(modules):
#                     if potential_prereq == module:
#                         cols[i+1].markdown("—", help="A module cannot be its own prerequisite")
#                     else:
#                         current_prereqs = st.session_state.prereqs.get(module, [])
#                         is_checked = potential_prereq in current_prereqs
                        
#                         checked = cols[i+1].checkbox(
#                             "",
#                             value=is_checked,
#                             key=f"prereq_{module}_{potential_prereq}",
#                             label_visibility="collapsed"
#                         )
                        
#                         # Update session state
#                         if checked and potential_prereq not in current_prereqs:
#                             st.session_state.prereqs[module].append(potential_prereq)
#                         elif not checked and potential_prereq in current_prereqs:
#                             st.session_state.prereqs[module].remove(potential_prereq)
            
#             # Add vertical space after table
#             st.markdown("<br>", unsafe_allow_html=True)
    
#     else:  # Individual Module Editor (default)
#         st.markdown("**Select prerequisites for each module individually**")
#         st.caption("💡 Tip: Start with foundational modules (M1-M4) before configuring advanced ones")
        
#         # Create expanders for each module
#         for module in modules:
#             with st.expander(f"{module} - {module_names[module]}"):
#                 # Get available prerequisites (all modules except itself)
#                 available_prereqs = [m for m in modules if m != module]
                
#                 # Format options with module names
#                 formatted_options = [f"{m} - {module_names[m]}" for m in available_prereqs]
                
#                 # Get current selections with proper formatting
#                 current_selections = st.session_state.prereqs.get(module, [])
#                 formatted_selections = [f"{m} - {module_names[m]}" for m in current_selections]
                
#                 # Multiselect with formatted display
#                 selected = st.multiselect(
#                     f"Prerequisites for {module}:",
#                     formatted_options,
#                     default=formatted_selections,
#                     key=f"multi_{module}"
#                 )
                
#                 # Convert back to module codes
#                 selected_codes = [opt.split(" - ")[0] for opt in selected]
#                 st.session_state.prereqs[module] = selected_codes
        
#         # Add validation button
#         if st.button("🔍 Validate Prerequisites"):
#             # Check for missing modules in prerequisites
#             invalid_prereqs = {}
#             for module, prereqs in st.session_state.prereqs.items():
#                 for prereq in prereqs:
#                     if prereq not in modules:
#                         invalid_prereqs.setdefault(module, []).append(prereq)
            
#             if invalid_prereqs:
#                 st.warning("⚠️ Found invalid prerequisites:")
#                 for module, invalids in invalid_prereqs.items():
#                     st.write(f"- {module}: {', '.join(invalids)}")
#                 st.info("These prerequisites will be ignored during scheduling")
#             else:
#                 st.success("✅ All prerequisites reference valid modules!")
            
#             # Check for circular dependencies
#             scheduler = CohortScheduler(
#                 modules=modules,
#                 prereqs=st.session_state.prereqs,
#                 cohort_starts={'C1': 1},  # Dummy cohort for validation
#                 config=SchedulerConfig(max_terms=10)
#             )
#             cycles = scheduler.detect_cycles()
#             if cycles:
#                 st.error("❌ Found circular dependencies:")
#                 for cycle in cycles:
#                     st.write(f"→ {' → '.join(cycle)} → {cycle[0]}")
#                 st.info("Fix these cycles before running the scheduler")
#             else:
#                 st.success("✅ No circular dependencies detected!")

#     # Display current configuration
#     st.divider()
#     st.subheader("Current Prerequisite Configuration")
    
#     config_display = []
#     for module in modules:
#         prereq_list = st.session_state.prereqs.get(module, [])
#         if prereq_list:
#             prereq_names = [f"{p} ({module_names[p]})" for p in prereq_list if p in module_names]
#             config_display.append(f"**{module} ({module_names[module]}):** {', '.join(prereq_names)}")
#         else:
#             config_display.append(f"**{module} ({module_names[module]}):** No Prerequisites")
    
#     st.markdown("\n\n".join(config_display))

# with tab2:
#     st.header("Run Scheduler")
    
#     # Configuration options
#     col1, col2 = st.columns(2)
#     with col1:
#         max_terms = st.number_input("Maximum Terms", min_value=10, max_value=200, value=50,
#                                   help="Maximum number of terms to consider for scheduling")
#     with col2:
#         modules_per_term = st.number_input("Max Modules per Cohort per Term", min_value=1, max_value=5, value=1,
#                                           help="How many modules a cohort can take in a single term")
    
#     # Cohort configuration
#     st.subheader("Cohort Start Terms")
#     st.caption("Configure when each cohort begins their studies")
    
#     cohort_starts = {}
#     cohort_cols = st.columns(4)
#     default_starts = [1, 2, 4, 6, 8, 10, 12, 14]
#     for i, cohort in enumerate([f'C{i+1}' for i in range(8)]):
#         with cohort_cols[i % 4]:
#             cohort_starts[cohort] = st.number_input(
#                 f"Start term for {cohort}",
#                 min_value=1,
#                 max_value=50,
#                 # value=i*2 + 1, #
#                 value=default_starts[i],
#                 key=f"cohort_{cohort}"
#             )
    
#     if st.button("🚀 Run Scheduler", type="primary", use_container_width=True):
#         with st.spinner("Running optimisation..."):
#             # Capture output
#             old_stdout = sys.stdout
#             sys.stdout = io.StringIO()
            
#             try:
#                 # Run scheduler
#                 config = SchedulerConfig(
#                     max_terms=max_terms,
#                     max_modules_per_cohort_per_term=modules_per_term,
#                     verbose=False
#                 )
                
#                 optimized = CohortScheduler(
#                     modules=modules,
#                     prereqs=st.session_state.prereqs,
#                     cohort_starts=cohort_starts,
#                     config=config
#                 )
                
#                 success = optimized.find_optimal_schedule()
#                 summary = optimized.get_schedule_summary()
                
#                 # Store results in session state
#                 st.session_state.scheduler = optimized
#                 st.session_state.summary = summary
#                 st.session_state.success = success
#                 st.session_state.cohort_starts = cohort_starts
                
#                 output = sys.stdout.getvalue()
#                 sys.stdout = old_stdout
                
#                 if success:
#                     st.success("✅ Scheduling completed successfully!")
#                 else:
#                     st.warning("⚠️ Scheduling completed with incomplete cohorts")
                
#                 # Show summary metrics
#                 col1, col2, col3 = st.columns(3)
#                 with col1:
#                     st.metric("Total Module Runs", summary['total_runs'])
#                 with col2:
#                     st.metric("Schedule Length", f"{summary['max_term']} terms")
#                 with col3:
#                     baseline_runs = 8 * 12  # 8 cohorts * 12 modules
#                     improvement = (baseline_runs - summary['total_runs']) / baseline_runs * 100
#                     st.metric("Improvement vs Baseline", f"{improvement:.1f}%")
                
#                 st.info("✨ Go to the 'Results' tab to view detailed schedule and download CSV files.")
                
#             except Exception as e:
#                 sys.stdout = old_stdout
#                 st.error(f"❌ Error running scheduler: {str(e)}")
#                 st.exception(e)

# with tab3:
#     st.header("Results & Downloads")
    
#     if 'scheduler' not in st.session_state:
#         st.info("👈 Please run the scheduler first in the 'Run Scheduler' tab.")
#         st.image("https://streamlit.io/images/hero.png", width=300, caption="Run scheduler to see results")
#     else:
#         scheduler = st.session_state.scheduler
#         summary = st.session_state.summary
#         cohort_starts = st.session_state.cohort_starts
        
#         # Download buttons
#         st.subheader("📥 Download CSV Files")
        
#         col1, col2, col3, col4 = st.columns(4)
        
#         # Generate CSVs
#         def generate_term_csv():
#             output = io.StringIO()
#             writer = csv.writer(output)
#             writer.writerow(['Term', 'Module_Code', 'Module_Name', 'Cohorts'])
#             for term in sorted(scheduler.schedule.keys()):
#                 for module, cohorts in scheduler.schedule[term].items():
#                     writer.writerow([term, module, module_names[module], ','.join(sorted(cohorts))])
#             return output.getvalue()
        
#         def generate_cohort_csv():
#             output = io.StringIO()
#             writer = csv.writer(output)
#             writer.writerow(['Cohort', 'Module_Code', 'Module_Name', 'Term_Taken', 'Start_Term'])
#             for cohort in sorted(scheduler.cohort_progress.keys()):
#                 start_term = cohort_starts[cohort]
#                 for module, term in sorted(scheduler.cohort_progress[cohort].items(), key=lambda x: x[1]):
#                     writer.writerow([cohort, module, module_names[module], term, start_term])
#             return output.getvalue()
        
#         def generate_metrics_csv():
#             output = io.StringIO()
#             writer = csv.writer(output)
#             writer.writerow(['Metric', 'Baseline', 'Optimized', 'Improvement_Percent'])
#             baseline_runs = 8 * 12  # 8 cohorts * 12 modules
#             opt_runs = summary['total_runs']
#             improvement = (baseline_runs - opt_runs) / baseline_runs * 100 if baseline_runs > 0 else 0
#             writer.writerow(['Total_Module_Runs', baseline_runs, opt_runs, f"{improvement:.2f}"])
#             writer.writerow(['Schedule_Length', 'N/A', summary['max_term'], 'N/A'])
#             return output.getvalue()
        
#         def generate_mapping_csv():
#             output = io.StringIO()
#             writer = csv.writer(output)
#             writer.writerow(['Module_Code', 'Module_Name', 'Terms_Offered', 'Run_Count'])
#             for module in sorted(module_names.keys()):
#                 terms = sorted(scheduler.module_runs[module])
#                 writer.writerow([module, module_names[module], ','.join(map(str, terms)), len(terms)])
#             return output.getvalue()
        
#         with col1:
#             st.download_button(
#                 "📄 Term Schedule",
#                 generate_term_csv(),
#                 "term_by_term_schedule.csv",
#                 "text/csv",
#                 help="Schedule showing which modules run each term and which cohorts attend"
#             )
        
#         with col2:
#             st.download_button(
#                 "📄 Cohort Progression",
#                 generate_cohort_csv(),
#                 "cohort_progression.csv",
#                 "text/csv",
#                 help="Complete progression of each cohort through all modules"
#             )
        
#         with col3:
#             st.download_button(
#                 "📊 Metrics",
#                 generate_metrics_csv(),
#                 "metrics_comparison.csv",
#                 "text/csv",
#                 help="Comparison of optimised schedule vs baseline (one module run per cohort)"
#             )
        
#         with col4:
#             st.download_button(
#                 "🔍 Module Mapping",
#                 generate_mapping_csv(),
#                 "module_term_mapping.csv",
#                 "text/csv",
#                 help="Mapping of which modules run in which terms"
#             )
        
#         # Display detailed results
#         st.divider()
#         # Term-by-Cohort Matrix View
#         st.subheader("📊 Term × Cohort Matrix View")
#         if not scheduler.schedule:
#             st.info("No schedule generated. Check for errors in prerequisites or configuration.")
#         else:
#             max_term = summary['max_term']
            
#             # Create data structure for the matrix
#             matrix_data = []
            
#             # Header row
#             header_row = ["Term","Date"]
#             for cohort in sorted(scheduler.cohorts):
#                 header_row.append(f"Cohort {cohort}")
#             matrix_data.append(header_row)
            
#             # Data rows for each term
#             for term in range(1, max_term + 1):
#                 row = [f"T{term}"]
#                 row.append(get_term_date(term))
#                 # For each cohort, find modules scheduled in this term
#                 for cohort in sorted(scheduler.cohorts):
#                     modules_in_term = []
#                     # Check all modules scheduled in this term
#                     if term in scheduler.schedule:
#                         for module, cohorts in scheduler.schedule[term].items():
#                             if cohort in cohorts:
#                                 modules_in_term.append(module_names[module])
                    
#                     if modules_in_term:
#                         # Join multiple modules with line breaks for better readability in cells
#                         row.append("\n".join(modules_in_term))
#                     else:
#                         row.append("")
#                 matrix_data.append(row)
            
#             # Convert to pandas DataFrame for better display
#             import pandas as pd
#             df = pd.DataFrame(matrix_data[1:], columns=matrix_data[0])
            
#             # Style the dataframe
#             styled_df = df.style.set_properties(**{
#                 'text-align': 'left',
#                 'white-space': 'pre-wrap',
#                 'vertical-align': 'top'
#             }).set_table_styles([
#                 {'selector': 'th', 'props': [('background-color', '#f0f2f6'), ('font-weight', 'bold')]},
#                 {'selector': 'td', 'props': [('border', '1px solid #e0e0e0'), ('padding', '8px')]},
#                 {'selector': 'tr:hover', 'props': [('background-color', '#f9f9f9')]}
#             ])
            
#             # Display the styled dataframe
#             st.dataframe(styled_df, height=600, use_container_width=True)
            
#             # Create CSV for spreadsheet download - Term × Cohort Matrix
#             def generate_matrix_csv():
#                 import csv
#                 import io
                
#                 output = io.StringIO()
#                 writer = csv.writer(output)
                
#                 # Write header row
#                 writer.writerow(["Term", "Date"] + [f"Cohort {c}" for c in sorted(scheduler.cohorts)])
                
#                 # Write data rows
#                 for term in range(1, max_term + 1):
#                     term_date = TERM_DATES[term]
#                     row = [f"T{term}", term_date]
#                     for cohort in sorted(scheduler.cohorts):
#                         modules_in_term = []
#                         if term in scheduler.schedule:
#                             for module, cohorts in scheduler.schedule[term].items():
#                                 if cohort in cohorts:
#                                     modules_in_term.append(module_names[module])
                        
#                         if modules_in_term:
#                             row.append("; ".join(modules_in_term))
#                         else:
#                             row.append("")
#                     writer.writerow(row)
                
#                 return output.getvalue()
#                     # Download button for term dates
#         def generate_term_dates_csv():
#                 output = io.StringIO()
#                 writer = csv.writer(output)
#                 writer.writerow(['Term', 'Date'])
#                 for term in range(1, 27):
#                     writer.writerow([f"Term {term}", get_term_date(term)])
#                 return output.getvalue()
#                 # Add download button for spreadsheet format
#         st.download_button(
#                     "📥 Download Term×Cohort Matrix (CSV)",
#                     generate_matrix_csv(),
#                     "term_cohort_matrix.csv",
#                     "text/csv",
#                     help="Download in spreadsheet format with terms as rows and cohorts as columns"
#                     )
#         # Module-term mapping - SHOW ALL MODULES BY DEFAULT
#         st.subheader("🗺️ Module-Term Mapping for All Modules")
        
#         # Create expanders for each module
#         for module in sorted(module_names.keys()):
#             with st.expander(f"{module_names[module]} ({module})"):
#                 if module in scheduler.module_runs:
#                     terms = sorted(scheduler.module_runs[module])
#                     if terms:
#                         st.markdown(f"**Offered in terms:** {', '.join(map(str, terms))}")
#                         st.markdown(f"**Total runs:** {len(terms)}")
                        
#                         # Visual timeline
#                         max_term = max(terms + [1])
#                         timeline = ["▢"] * (max_term + 1)
#                         for t in terms:
#                             if t <= max_term:
#                                 timeline[t] = "✅"
#                         timeline_str = "".join(timeline[1:])
#                         st.markdown(f"**Term timeline:** `1`{''.join(timeline[1:])}`{max_term}`")
#                     else:
#                         st.info(f"Module {module} is never scheduled")
#                 else:
#                     st.warning(f"Module {module} not found in schedule")    
#         # Term-by-term schedule - REVISED FORMAT
#         st.subheader("📅 Term-by-Term Schedule (Detailed View)") 
        
#         if not scheduler.schedule:
#             st.info("No schedule generated. Check for errors in prerequisites or configuration.")
#         else:
#             max_term = summary['max_term']
#             schedule_lines = []
            
#             # Header
#             schedule_lines.append("="*80)
#             schedule_lines.append("DETAILED OPTIMISED SCHEDULE (Term-by-Term)")
#             schedule_lines.append("="*80)
            
#             # Generate schedule lines for all terms from 1 to max_term
#             for term in range(1, max_term + 1):
#                 if term in scheduler.schedule and scheduler.schedule[term]:
#                     modules_in_term = []
#                     # Sort modules by module code for consistent ordering
#                     for module in sorted(scheduler.schedule[term].keys()):
#                         cohorts = sorted(scheduler.schedule[term][module])
#                         cohort_str = ', '.join(cohorts)
#                         modules_in_term.append(f"{module_names[module]} ({cohort_str})")
                    
#                     # Join modules with tab separation
#                     term_line = f"T{term}: " + "\t".join(modules_in_term)
#                     schedule_lines.append(term_line)
#                 else:
#                     # For empty terms, show placeholder
#                     schedule_lines.append(f"T{term}: (no modules scheduled)")
            
#             # Footer
#             schedule_lines.append("="*80)
            
#             # Display as monospace text
#             schedule_text = "\n".join(schedule_lines)
#             st.text(schedule_text)
            
#             # Create CSV for spreadsheet download - Term-by-Term
#             def generate_term_csv_spreadsheet():
#                 import csv
#                 import io
                
#                 output = io.StringIO()
#                 writer = csv.writer(output)
                
#                 # Write header row
#                 writer.writerow(['Term', 'Module 1', 'Module 2', 'Module 3', 'Module 4', 'Module 5', 'Module 6', 'Module 7', 'Module 8', 'Module 9', 'Module 10', 'Module 11', 'Module 12'])
                
#                 # Write data rows
#                 for term in range(1, max_term + 1):
#                     row = [f"T{term}"]
#                     if term in scheduler.schedule and scheduler.schedule[term]:
#                         # Sort modules by module code
#                         for module in sorted(scheduler.schedule[term].keys()):
#                             cohorts = sorted(scheduler.schedule[term][module])
#                             cohort_str = ', '.join(cohorts)
#                             row.append(f"{module_names[module]} ({cohort_str})")
#                     writer.writerow(row)
                
#                 return output.getvalue()
            
#             # Add download button for spreadsheet format
#             st.download_button(
#                 "📊 Download Term Schedule (CSV)",
#                 generate_term_csv_spreadsheet(),
#                 "term_schedule_spreadsheet.csv",
#                 "text/csv",
#                 help="Download in spreadsheet format with terms as rows and modules as columns"
#             )
        
#         # Cohort progression - REVISED HORIZONTAL FORMAT
#         st.subheader("👥 Cohort Progression (Horizontal View)")
#         if not scheduler.cohort_progress:
#             st.info("No cohort progression data available.")
#         else:
#             cohort_lines = []
            
#             # Header
#             cohort_lines.append("="*80)
#             cohort_lines.append("PER-COHORT MODULE PROGRESSION")
#             cohort_lines.append("="*80)
            
#             # Generate progression lines for all cohorts
#             for cohort in sorted(scheduler.cohort_progress.keys()):
#                 if cohort in scheduler.cohort_progress:
#                     modules_in_cohort = []
#                     # Sort modules by term taken
#                     sorted_modules = sorted(
#                         scheduler.cohort_progress[cohort].items(),
#                         key=lambda x: x[1]
#                     )
                    
#                     for module, term in sorted_modules:
#                         modules_in_cohort.append(f"{module_names[module]}(T{term})")
                    
#                     # Join modules with tab separation
#                     cohort_line = f"{cohort}: " + "\t".join(modules_in_cohort)
#                     cohort_lines.append(cohort_line)
#                 else:
#                     cohort_lines.append(f"{cohort}: (no modules scheduled)")
            
#             # Footer
#             cohort_lines.append("="*80)
            
#             # Display as monospace text
#             cohort_text = "\n".join(cohort_lines)
#             st.text(cohort_text)
            
#             # Create CSV for spreadsheet download - Cohort Progression
#             def generate_cohort_csv_spreadsheet():
#                 import csv
#                 import io
                
#                 output = io.StringIO()
#                 writer = csv.writer(output)
                
#                 # Write header row
#                 writer.writerow(['Cohort', 'Module 1', 'Module 2', 'Module 3', 'Module 4', 'Module 5', 'Module 6', 'Module 7', 'Module 8', 'Module 9', 'Module 10', 'Module 11', 'Module 12'])
                
#                 # Write data rows
#                 for cohort in sorted(scheduler.cohort_progress.keys()):
#                     row = [cohort]
#                     if cohort in scheduler.cohort_progress:
#                         # Sort modules by term taken
#                         sorted_modules = sorted(
#                             scheduler.cohort_progress[cohort].items(),
#                             key=lambda x: x[1]
#                         )
#                         for module, term in sorted_modules:
#                             row.append(f"{module_names[module]}(T{term})")
#                     writer.writerow(row)
                
#                 return output.getvalue()
            
#             # Add download button for spreadsheet format
#             st.download_button(
#                 "📊 Download Cohort Progression (CSV)",
#                 generate_cohort_csv_spreadsheet(),
#                 "cohort_progression_spreadsheet.csv",
#                 "text/csv",
#                 help="Download in spreadsheet format with cohorts as rows and modules as columns"
#             )  
            

#             st.caption("💡 Tip: This view shows which modules each cohort takes in each term. Hover over cells to see full content, or download the CSV for complete details.")       


# # Footer
# st.divider()
# st.markdown("""
# **💡 Tips for Better Schedules:**
# - Start with foundational modules having no prerequisites
# - Avoid circular dependencies (A requires B, B requires A)
# - Increase 'Max Modules per Cohort per Term' for faster completion
# - Adjust cohort start terms to balance resource usage
# - Use presets as starting points for your configuration
# """)

# st.caption("Scheduler v2.1 • Handles 12 modules and 8 cohorts • Uses greedy optimisation algorithm")





