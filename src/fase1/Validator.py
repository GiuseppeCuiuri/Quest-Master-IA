import re
from typing import List, Dict, Set, Tuple

import networkx as nx

from src.fase1.PDDLGenerator import LoreDocument


class PDDLConstraintsValidator:
    """Valida che il PDDL generato rispetti i constraints narrativi"""

    def __init__(self, domain_pddl: str, problem_pddl: str, lore: LoreDocument):
        self.domain_pddl = domain_pddl
        self.problem_pddl = problem_pddl
        self.lore = lore
        self.validation_errors = []
        self.validation_warnings = []

    def validate_all(self) -> Tuple[bool, List[str], List[str]]:
        """Esegue tutte le validazioni"""
        self.validation_errors = []
        self.validation_warnings = []

        # Validazioni strutturali
        self._validate_branching_factor()
        self._validate_depth_constraints()
        self._validate_narrative_consistency()
        self._validate_goal_achievability()
        self._validate_action_diversity()

        # Validazioni semantiche
        self._validate_logical_dependencies()
        self._validate_no_dead_ends()
        self._validate_narrative_flow()

        is_valid = len(self.validation_errors) == 0
        return is_valid, self.validation_errors, self.validation_warnings

    def _validate_branching_factor(self):
        """Verifica che il branching factor sia rispettato"""
        # Estrai tutte le azioni dal domain
        action_pattern = r':action\s+(\S+)'
        actions = re.findall(action_pattern, self.domain_pddl)

        # Costruisci grafo delle possibili transizioni
        state_graph = self._build_state_graph(actions)

        # Verifica branching per ogni stato
        min_branch, max_branch = self.lore.branching_factor

        for node in state_graph.nodes():
            out_degree = state_graph.out_degree(node)
            if out_degree < min_branch:
                self.validation_warnings.append(
                    f"Stato {node} ha solo {out_degree} azioni possibili (minimo richiesto: {min_branch})"
                )
            elif out_degree > max_branch:
                self.validation_warnings.append(
                    f"Stato {node} ha {out_degree} azioni possibili (massimo consentito: {max_branch})"
                )

    def _validate_depth_constraints(self):
        """Verifica che la profondità della quest sia nei limiti"""
        # Estrai stato iniziale e goal
        initial_state = self._extract_initial_state()
        goal_state = self._extract_goal_state()

        # Simula ricerca del percorso minimo/massimo
        min_depth, max_depth = self.lore.depth_constraints

        # Questa è una validazione semplificata
        # In produzione, useresti il planner per calcolare i percorsi reali
        estimated_min_steps = len(goal_state) // 2  # Stima ottimistica
        estimated_max_steps = len(goal_state) * 3  # Stima pessimistica

        if estimated_min_steps < min_depth:
            self.validation_warnings.append(
                f"La quest potrebbe essere completata in {estimated_min_steps} passi (minimo richiesto: {min_depth})"
            )

        if estimated_max_steps > max_depth:
            self.validation_warnings.append(
                f"La quest potrebbe richiedere fino a {estimated_max_steps} passi (massimo consentito: {max_depth})"
            )

    def _validate_narrative_consistency(self):
        """Verifica la coerenza narrativa tra Lore e PDDL"""
        # Verifica che tutti gli elementi del Lore siano nel PDDL
        lore_elements = {
            'characters': set(self.lore.characters),
            'locations': set(self.lore.locations),
            'items': set(self.lore.items),
            'obstacles': set(self.lore.obstacles)
        }

        pddl_elements = self._extract_pddl_objects()

        for element_type, lore_set in lore_elements.items():
            pddl_set = pddl_elements.get(element_type, set())

            # Normalizza i nomi per il confronto
            normalized_lore = {self._normalize_name(e) for e in lore_set}
            normalized_pddl = {self._normalize_name(e) for e in pddl_set}

            missing = normalized_lore - normalized_pddl
            if missing:
                self.validation_errors.append(
                    f"Elementi {element_type} mancanti nel PDDL: {missing}"
                )

            extra = normalized_pddl - normalized_lore
            if extra:
                self.validation_warnings.append(
                    f"Elementi {element_type} extra nel PDDL non presenti nel Lore: {extra}"
                )

    def _validate_goal_achievability(self):
        """Verifica che il goal sia teoricamente raggiungibile"""
        goal_predicates = self._extract_goal_state()
        available_effects = self._extract_all_effects()

        for goal_pred in goal_predicates:
            # Semplifica il predicato per il matching
            simplified_goal = re.sub(r'\?[a-zA-Z0-9_]+', '?var', goal_pred)

            can_achieve = False
            for effect in available_effects:
                simplified_effect = re.sub(r'\?[a-zA-Z0-9_]+', '?var', effect)
                if simplified_goal in simplified_effect:
                    can_achieve = True
                    break

            if not can_achieve:
                self.validation_errors.append(
                    f"Goal '{goal_pred}' non può essere raggiunto da nessuna azione"
                )

    def _validate_action_diversity(self):
        """Verifica che le azioni siano sufficientemente diverse"""
        actions = self._extract_actions()

        # Calcola similarità tra azioni
        similarity_threshold = 0.8
        too_similar = []

        for i, action1 in enumerate(actions):
            for j, action2 in enumerate(actions[i + 1:], i + 1):
                similarity = self._calculate_action_similarity(action1, action2)
                if similarity > similarity_threshold:
                    too_similar.append((action1['name'], action2['name'], similarity))

        if too_similar:
            for a1, a2, sim in too_similar[:3]:  # Mostra solo prime 3
                self.validation_warnings.append(
                    f"Azioni '{a1}' e '{a2}' sono troppo simili ({sim:.2%})"
                )

    def _validate_logical_dependencies(self):
        """Verifica che le dipendenze logiche siano sensate"""
        actions = self._extract_actions()

        for action in actions:
            preconds = action.get('preconditions', [])
            effects = action.get('effects', [])

            # Verifica che non ci siano contraddizioni immediate
            for precond in preconds:
                if precond.startswith('(not '):
                    positive = precond[5:-1]
                    if positive in effects:
                        self.validation_errors.append(
                            f"Azione '{action['name']}' richiede NOT {positive} ma lo produce come effetto"
                        )

    def _validate_no_dead_ends(self):
        """Verifica che non ci siano stati senza uscita"""
        # Questa è una validazione semplificata
        # In produzione, costruiresti il grafo degli stati completo

        locations = [loc for loc in self._extract_pddl_objects().get('locations', [])]
        connections = self._extract_connections()

        # Verifica che ogni location sia raggiungibile
        if locations:
            graph = nx.Graph()
            graph.add_nodes_from(locations)
            graph.add_edges_from(connections)

            if not nx.is_connected(graph):
                disconnected = list(nx.connected_components(graph))
                self.validation_errors.append(
                    f"Il mondo non è completamente connesso. Componenti separate: {len(disconnected)}"
                )

    def _validate_narrative_flow(self):
        """Verifica che il flusso narrativo sia logico"""
        # Verifica che ci sia progressione narrativa
        initial_predicates = set(self._extract_initial_state())
        goal_predicates = set(self._extract_goal_state())

        # Dovrebbero esserci differenze significative
        if len(goal_predicates - initial_predicates) < 3:
            self.validation_warnings.append(
                "Il goal è troppo simile allo stato iniziale. Considera di aggiungere più obiettivi."
            )

        # Verifica presenza di elementi narrativi chiave
        has_character_interaction = any('talked-to' in g for g in goal_predicates)
        has_exploration = any('explored' in g for g in goal_predicates)
        has_item_usage = any('used' in g or 'has' in g for g in goal_predicates)

        if not has_character_interaction:
            self.validation_warnings.append(
                "Nessuna interazione con personaggi richiesta nel goal"
            )

        if not has_exploration and len(self.lore.locations) > 2:
            self.validation_warnings.append(
                "Nessuna esplorazione richiesta nonostante multiple location"
            )

    # Metodi di supporto
    def _build_state_graph(self, actions: List[str]) -> nx.DiGraph:
        """Costruisce grafo semplificato degli stati"""
        graph = nx.DiGraph()

        # Aggiungi nodi per ogni location
        locations = self._extract_pddl_objects().get('locations', [])
        for loc in locations:
            graph.add_node(loc)

        # Aggiungi archi basati sulle azioni
        for action in actions:
            if 'move' in action:
                # Semplificazione: assume che move connetta location
                for loc1 in locations:
                    for loc2 in locations:
                        if loc1 != loc2:
                            graph.add_edge(loc1, loc2, action=action)

        return graph

    def _extract_initial_state(self) -> List[str]:
        """Estrae predicati dello stato iniziale"""
        init_pattern = r':init\s*\((.*?)\)\s*\)'
        match = re.search(init_pattern, self.problem_pddl, re.DOTALL)
        if match:
            predicates = re.findall(r'\([^)]+\)', match.group(1))
            return [p.strip() for p in predicates]
        return []

    def _extract_goal_state(self) -> List[str]:
        """Estrae predicati del goal"""
        goal_pattern = r':goal\s*\(\s*and\s*(.*?)\)\s*\)'
        match = re.search(goal_pattern, self.problem_pddl, re.DOTALL)
        if match:
            predicates = re.findall(r'\([^)]+\)', match.group(1))
            return [p.strip() for p in predicates]
        return []

    def _extract_pddl_objects(self) -> Dict[str, Set[str]]:
        """Estrae tutti gli oggetti dal problem PDDL"""
        objects = {
            'characters': set(),
            'locations': set(),
            'items': set(),
            'obstacles': set()
        }

        objects_pattern = r':objects\s*(.*?)(?=:init|:goal|\))'
        match = re.search(objects_pattern, self.problem_pddl, re.DOTALL)

        if match:
            objects_text = match.group(1)
            # Pattern per ogni tipo di oggetto
            patterns = {
                'characters': r'(\S+)\s*-\s*character',
                'locations': r'(\S+)\s*-\s*location',
                'items': r'(\S+)\s*-\s*item',
                'obstacles': r'(\S+)\s*-\s*obstacle'
            }

            for obj_type, pattern in patterns.items():
                matches = re.findall(pattern, objects_text)
                objects[obj_type] = set(matches)

        return objects

    def _extract_all_effects(self) -> List[str]:
        """Estrae tutti gli effetti possibili dalle azioni"""
        effects = []
        action_pattern = r':action.*?:effect\s*\((.*?)\)\s*\)'
        matches = re.findall(action_pattern, self.domain_pddl, re.DOTALL)

        for match in matches:
            # Estrai predicati individuali
            predicates = re.findall(r'\([^)]+\)', match)
            effects.extend(predicates)

        return effects

    def _extract_actions(self) -> List[Dict[str, any]]:
        """Estrae informazioni dettagliate su ogni azione"""
        actions = []
        action_pattern = r':action\s+(\S+)(.*?)(?=:action|\Z)'
        matches = re.findall(action_pattern, self.domain_pddl, re.DOTALL)

        for name, content in matches:
            action = {'name': name}

            # Estrai precondizioni
            precond_match = re.search(r':precondition\s*\((.*?)\)', content, re.DOTALL)
            if precond_match:
                action['preconditions'] = re.findall(r'\([^)]+\)', precond_match.group(1))

            # Estrai effetti
            effect_match = re.search(r':effect\s*\((.*?)\)', content, re.DOTALL)
            if effect_match:
                action['effects'] = re.findall(r'\([^)]+\)', effect_match.group(1))

            actions.append(action)

        return actions

    def _calculate_action_similarity(self, action1: Dict, action2: Dict) -> float:
        """Calcola similarità tra due azioni"""
        # Similarità basata su precondizioni ed effetti condivisi
        preconds1 = set(action1.get('preconditions', []))
        preconds2 = set(action2.get('preconditions', []))
        effects1 = set(action1.get('effects', []))
        effects2 = set(action2.get('effects', []))

        if not (preconds1 or preconds2 or effects1 or effects2):
            return 0.0

        # Jaccard similarity
        precond_similarity = len(preconds1 & preconds2) / len(preconds1 | preconds2) if (preconds1 | preconds2) else 0
        effect_similarity = len(effects1 & effects2) / len(effects1 | effects2) if (effects1 | effects2) else 0

        return (precond_similarity + effect_similarity) / 2

    def _extract_connections(self) -> List[Tuple[str, str]]:
        """Estrae connessioni tra location"""
        connections = []
        init_state = self._extract_initial_state()

        for predicate in init_state:
            if 'connected' in predicate:
                # Estrai le due location
                match = re.search(r'connected\s+(\S+)\s+(\S+)', predicate)
                if match:
                    connections.append((match.group(1), match.group(2)))

        return connections

    def _normalize_name(self, name: str) -> str:
        """Normalizza i nomi per confronto"""
        return re.sub(r'[^a-z0-9]', '', name.lower())