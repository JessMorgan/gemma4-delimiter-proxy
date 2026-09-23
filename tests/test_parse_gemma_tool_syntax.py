"""Unit tests for parse_gemma_tool_syntax."""

import json

from gemma4_delimiter_proxy.proxy import parse_gemma_tool_syntax


def test_quoted_args():
    func_name, args_str = parse_gemma_tool_syntax('call:my_func{"arg1": 23}')
    assert func_name == "my_func"
    assert json.loads(args_str) == {"arg1": 23}


def test_unquoted_keys():
    func_name, args_str = parse_gemma_tool_syntax('call:get_weather{location: "Boston"}')
    assert func_name == "get_weather"
    assert json.loads(args_str) == {"location": "Boston"}


def test_escape_sequence():
    func_name, args_str = parse_gemma_tool_syntax('call:get_weather{location:<|"|>Boston<|"|>}')
    assert func_name == "get_weather"
    assert json.loads(args_str) == {"location": "Boston"}


def test_combined_unquoted_and_escaped():
    func_name, args_str = parse_gemma_tool_syntax(
        'call:search{query: <|"|>hello world<|"|>, limit: 5}'
    )
    assert func_name == "search"
    assert json.loads(args_str) == {"query": "hello world", "limit": 5}


def test_quoted_value_with_inner_key_colon_unchanged():
    # Regression: the key-quoting repair must not match inside quoted strings
    func_name, args_str = parse_gemma_tool_syntax(
        'call:note_it{"note": "see {time: 5}", level: 2}'
    )
    assert func_name == "note_it"
    assert json.loads(args_str) == {"note": "see {time: 5}", "level": 2}


def test_escaped_quote_inside_string():
    func_name, args_str = parse_gemma_tool_syntax(
        'call:save{"text": "a \\"quoted\\" {fake: 1} string", id: 7}'
    )
    assert func_name == "save"
    assert json.loads(args_str) == {"text": 'a "quoted" {fake: 1} string', "id": 7}


def test_no_function_name_falls_back():
    func_name, args_str = parse_gemma_tool_syntax("garbage without call syntax")
    assert func_name == "unknown_function"
    assert args_str == "{}"
