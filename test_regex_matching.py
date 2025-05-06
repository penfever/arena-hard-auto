#!/usr/bin/env python3
"""
Test script to verify regex pattern matching for judge outputs.
Specifically tests patterns from judge_config_multipattern.yaml for proper matching.
"""

import re
import yaml
import sys
import os
from pprint import pprint

def load_patterns(config_file):
    """Load regex patterns from config file"""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    patterns = []
    if "regex_patterns" in config and config["regex_patterns"]:
        for pattern_config in config["regex_patterns"]:
            pattern_obj = {
                "name": pattern_config["name"],
                "pattern": re.compile(pattern_config["pattern"]),
                "pattern_str": pattern_config["pattern"]
            }
            patterns.append(pattern_obj)
    return patterns

def get_score(judgment, patterns):
    """Simplified version of get_score from gen_judgment.py"""
    scores = {}
    continue_flag = False
    
    # Process each pattern
    for pattern_obj in patterns:
        pattern_name = pattern_obj['name']
        pattern = pattern_obj['pattern']
        
        print(f"\nTesting pattern '{pattern_name}': {pattern_obj['pattern_str']}")
        matches = pattern.findall(judgment)
        print(f"Matches found: {matches}")
        
        matches = [m for m in matches if m != ""]
        
        if len(set(matches)) == 0:
            # No matches for this pattern, continue requesting more tokens
            print(f"No matches found for pattern '{pattern_name}'")
            continue_flag = True
        elif len(set(matches)) == 1:
            # Single match for this pattern
            match = matches[0].strip("\n")
            print(f"Found match: '{match}'")
            scores[pattern_name] = match
        else:
            # Multiple different matches, this is invalid for this pattern
            print(f"Multiple different matches found: {matches}")
            scores[pattern_name] = None
    
    return scores, continue_flag

def test_problematic_string():
    """Test the problematic string with the regex pattern"""
    problematic_string = """Therefore, the final verdict is:

Assistant B is slightly better: [[B>A]]"""
    
    print("Testing problematic string:")
    print("-" * 50)
    print(repr(problematic_string))
    print("-" * 50)
    
    # Direct regex test
    direct_pattern = re.compile(r'\[\[([AB<>=]+)\]\]')
    direct_match = direct_pattern.findall(problematic_string)
    print(f"Direct regex test result: {direct_match}")
    
    # Load and test with patterns from config
    config_file = os.path.join("config", "judge_config_multipattern.yaml")
    try:
        patterns = load_patterns(config_file)
        print(f"Loaded {len(patterns)} patterns from {config_file}")
        
        # Test the string with each pattern
        scores, continue_flag = get_score(problematic_string, patterns)
        
        print("\nFinal scores:")
        pprint(scores)
        print(f"Continue flag: {continue_flag}")
        
        if "overall" in scores and scores["overall"]:
            print("\n✅ SUCCESS: The overall pattern matched correctly")
        else:
            print("\n❌ FAILURE: The overall pattern did not match")
            
            # Try various transformations to debug
            print("\nDebugging transformations:")
            transformations = [
                ("Strip whitespace", problematic_string.strip()),
                ("Replace newlines with spaces", problematic_string.replace("\n", " ")),
                ("Replace newlines with \\n", repr(problematic_string)),
                ("Force ASCII-only", problematic_string.encode('ascii', 'ignore').decode())
            ]
            
            for name, transformed_str in transformations:
                print(f"\n{name}:")
                print(transformed_str)
                direct_match = direct_pattern.findall(transformed_str)
                print(f"Match result: {direct_match}")
        
    except Exception as e:
        print(f"Error loading patterns: {e}")
        return False
    
    return "overall" in scores and scores["overall"]

def test_markdown_formatted_string():
    """Test a string with markdown formatting that should still match"""
    markdown_string = """My verdicts are as follows: **Correctness**: ((A>>B)). **Completeness**: ((A=B)). **Safety**: ((A=B)). **Conciseness**: ((A=B)). **Style**: ((A=B)). My final verdict is tie: [[A=B]]"""
    
    print("\n\nTesting markdown formatted string:")
    print("-" * 50)
    print(repr(markdown_string))
    print("-" * 50)
    
    # Load and test with patterns from config
    config_file = os.path.join("config", "judge_config_multipattern.yaml")
    try:
        patterns = load_patterns(config_file)
        
        # Test the string with each pattern
        scores, continue_flag = get_score(markdown_string, patterns)
        
        print("\nFinal scores:")
        pprint(scores)
        
        success = True
        for key in ["overall", "correctness", "completeness", "safety", "conciseness", "style"]:
            if key not in scores or not scores[key]:
                print(f"\n❌ FAILURE: The {key} pattern did not match")
                success = False
        
        if success:
            print("\n✅ SUCCESS: All patterns matched correctly")
        
    except Exception as e:
        print(f"Error loading patterns: {e}")
        return False
    
    return success

if __name__ == "__main__":
    print("=" * 60)
    print(" REGEX PATTERN MATCHING TEST ")
    print("=" * 60)
    
    test1_result = test_problematic_string()
    test2_result = test_markdown_formatted_string()
    
    if test1_result and test2_result:
        print("\nAll tests passed successfully! ✅")
        sys.exit(0)
    else:
        print("\nSome tests failed! ❌")
        sys.exit(1)