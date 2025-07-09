import json
from typing import Union

import yaml

from src.fase1.PDDLGenerator import LoreDocument


class LoreParser:
    """Parser avanzato per documenti Lore con supporto multi-formato"""

    @staticmethod
    def parse_lore_document(lore_input: Union[str, dict]) -> LoreDocument:
        """
        Parsa il documento Lore da vari formati (testo, JSON, YAML)
        """
        if isinstance(lore_input, dict):
            return LoreParser._parse_from_dict(lore_input)

        # Tenta di identificare il formato
        lore_text = lore_input.strip()

        # Prova JSON
        if lore_text.startswith('{'):
            try:
                data = json.loads(lore_text)
                return LoreParser._parse_from_dict(data)
            except json.JSONDecodeError:
                pass

        # Prova YAML
        if ':' in lore_text and not lore_text.startswith('QUEST:'):
            try:
                data = yaml.safe_load(lore_text)
                return LoreParser._parse_from_dict(data)
            except yaml.YAMLError:
                pass

        # Fallback al parser testuale migliorato
        return LoreParser._parse_from_text(lore_text)

    @staticmethod
    def _parse_from_dict(data: dict) -> LoreDocument:
        """Crea LoreDocument da dizionario"""
        return LoreDocument(
            quest_description=data.get('quest_description', ''),
            initial_state=data.get('initial_state', ''),
            goal_state=data.get('goal_state', ''),
            world_context=data.get('world_context', ''),
            characters=data.get('characters', []),
            locations=data.get('locations', []),
            items=data.get('items', []),
            obstacles=data.get('obstacles', []),
            branching_factor=tuple(data.get('branching_factor', [2, 4])),
            depth_constraints=tuple(data.get('depth_constraints', [3, 10]))
        )

    @staticmethod
    def _parse_from_text(lore_text: str) -> LoreDocument:
        """Parser testuale migliorato con regex e NLP base"""
        import re

        # Pattern più flessibili
        patterns = {
            'quest': r'(?:QUEST|Quest|quest):\s*(.+?)(?=(?:INITIAL|GOAL|WORLD|CHARACTERS|LOCATIONS|ITEMS|OBSTACLES|BRANCHING|DEPTH)|$)',
            'initial': r'(?:INITIAL|Initial|initial):\s*(.+?)(?=(?:QUEST|GOAL|WORLD|CHARACTERS|LOCATIONS|ITEMS|OBSTACLES|BRANCHING|DEPTH)|$)',
            'goal': r'(?:GOAL|Goal|goal):\s*(.+?)(?=(?:QUEST|INITIAL|WORLD|CHARACTERS|LOCATIONS|ITEMS|OBSTACLES|BRANCHING|DEPTH)|$)',
            'world': r'(?:WORLD|World|world):\s*(.+?)(?=(?:QUEST|INITIAL|GOAL|CHARACTERS|LOCATIONS|ITEMS|OBSTACLES|BRANCHING|DEPTH)|$)',
            'characters': r'(?:CHARACTERS|Characters|characters):\s*(.+?)(?=(?:QUEST|INITIAL|GOAL|WORLD|LOCATIONS|ITEMS|OBSTACLES|BRANCHING|DEPTH)|$)',
            'locations': r'(?:LOCATIONS|Locations|locations):\s*(.+?)(?=(?:QUEST|INITIAL|GOAL|WORLD|CHARACTERS|ITEMS|OBSTACLES|BRANCHING|DEPTH)|$)',
            'items': r'(?:ITEMS|Items|items):\s*(.+?)(?=(?:QUEST|INITIAL|GOAL|WORLD|CHARACTERS|LOCATIONS|OBSTACLES|BRANCHING|DEPTH)|$)',
            'obstacles': r'(?:OBSTACLES|Obstacles|obstacles):\s*(.+?)(?=(?:QUEST|INITIAL|GOAL|WORLD|CHARACTERS|LOCATIONS|ITEMS|BRANCHING|DEPTH)|$)',
            'branching': r'(?:BRANCHING|Branching|branching):\s*(\d+)\s*-\s*(\d+)',
            'depth': r'(?:DEPTH|Depth|depth):\s*(\d+)\s*-\s*(\d+)'
        }

        # Estrai informazioni con regex
        quest_description = re.search(patterns['quest'], lore_text, re.DOTALL)
        initial_state = re.search(patterns['initial'], lore_text, re.DOTALL)
        goal_state = re.search(patterns['goal'], lore_text, re.DOTALL)
        world_context = re.search(patterns['world'], lore_text, re.DOTALL)

        # Estrai liste
        characters = re.search(patterns['characters'], lore_text, re.DOTALL)
        locations = re.search(patterns['locations'], lore_text, re.DOTALL)
        items = re.search(patterns['items'], lore_text, re.DOTALL)
        obstacles = re.search(patterns['obstacles'], lore_text, re.DOTALL)

        # Estrai constraints
        branching = re.search(patterns['branching'], lore_text)
        depth = re.search(patterns['depth'], lore_text)

        # Processa i risultati
        def clean_text(match):
            return match.group(1).strip() if match else ""

        def parse_list(match):
            if not match:
                return []
            text = match.group(1).strip()
            # Supporta sia virgole che newline come separatori
            if ',' in text:
                return [item.strip() for item in text.split(',') if item.strip()]
            else:
                return [item.strip() for item in text.split('\n') if item.strip()]

        return LoreDocument(
            quest_description=clean_text(quest_description),
            initial_state=clean_text(initial_state),
            goal_state=clean_text(goal_state),
            world_context=clean_text(world_context),
            characters=parse_list(characters),
            locations=parse_list(locations),
            items=parse_list(items),
            obstacles=parse_list(obstacles),
            branching_factor=(
                int(branching.group(1)) if branching else 2,
                int(branching.group(2)) if branching else 4
            ),
            depth_constraints=(
                int(depth.group(1)) if depth else 3,
                int(depth.group(2)) if depth else 10
            )
        )