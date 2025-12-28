"""Unit tests for TextRouterConfig."""

import path_setup

path_setup.add_src_path()


import unittest


class TestTextRouterConfigDefaults(unittest.TestCase):
    """Tests TextRouterConfig default values."""

    def test_default_prefix(self) -> None:
        """Default prefix is '/text'."""
        from ner_controller.api.configs.text_router_config import TextRouterConfig

        config = TextRouterConfig()

        self.assertEqual(config.prefix, "/text")

    def test_default_tags(self) -> None:
        """Default tags is ('text-processing',)."""
        from ner_controller.api.configs.text_router_config import TextRouterConfig

        config = TextRouterConfig()

        self.assertEqual(config.tags, ("text-processing",))

    def test_config_with_defaults(self) -> None:
        """Can create config with all default values."""
        from ner_controller.api.configs.text_router_config import TextRouterConfig

        config = TextRouterConfig()

        self.assertEqual(config.prefix, "/text")
        self.assertEqual(len(config.tags), 1)
        self.assertIn("text-processing", config.tags)


class TestTextRouterConfigCustomValues(unittest.TestCase):
    """Tests TextRouterConfig with custom values."""

    def test_custom_prefix(self) -> None:
        """Can create config with custom prefix."""
        from ner_controller.api.configs.text_router_config import TextRouterConfig

        config = TextRouterConfig(prefix="/api/v1/text")

        self.assertEqual(config.prefix, "/api/v1/text")

    def test_custom_tags(self) -> None:
        """Can create config with custom tags."""
        from ner_controller.api.configs.text_router_config import TextRouterConfig

        config = TextRouterConfig(tags=("nlp", "text", "ner"))

        self.assertEqual(len(config.tags), 3)
        self.assertIn("nlp", config.tags)
        self.assertIn("text", config.tags)
        self.assertIn("ner", config.tags)

    def test_custom_prefix_and_tags(self) -> None:
        """Can create config with custom prefix and tags."""
        from ner_controller.api.configs.text_router_config import TextRouterConfig

        config = TextRouterConfig(
            prefix="/custom/text", tags=("custom-tag", "another-tag")
        )

        self.assertEqual(config.prefix, "/custom/text")
        self.assertEqual(len(config.tags), 2)


class TestTextRouterConfigImmutability(unittest.TestCase):
    """Tests TextRouterConfig frozen dataclass immutability."""

    def test_frozen_dataclass_prevents_field_reassignment(self) -> None:
        """Cannot reassign fields after instantiation."""
        from ner_controller.api.configs.text_router_config import TextRouterConfig

        config = TextRouterConfig()

        with self.assertRaises(Exception):  # FrozenInstanceError
            config.prefix = "/modified"

    def test_frozen_dataclass_prevents_field_addition(self) -> None:
        """Cannot add new fields after instantiation."""
        from ner_controller.api.configs.text_router_config import TextRouterConfig

        config = TextRouterConfig()

        with self.assertRaises(Exception):  # FrozenInstanceError
            config.new_field = "value"


class TestTextRouterConfigEquality(unittest.TestCase):
    """Tests TextRouterConfig equality."""

    def test_two_configs_with_same_values_are_equal(self) -> None:
        """Two instances with same values are equal."""
        from ner_controller.api.configs.text_router_config import TextRouterConfig

        config1 = TextRouterConfig(prefix="/text", tags=("tag1", "tag2"))
        config2 = TextRouterConfig(prefix="/text", tags=("tag1", "tag2"))

        self.assertEqual(config1, config2)

    def test_two_configs_with_different_values_not_equal(self) -> None:
        """Two instances with different values are not equal."""
        from ner_controller.api.configs.text_router_config import TextRouterConfig

        config1 = TextRouterConfig(prefix="/text")
        config2 = TextRouterConfig(prefix="/api/v1/text")

        self.assertNotEqual(config1, config2)

    def test_config_not_equal_to_other_type(self) -> None:
        """TextRouterConfig is not equal to other types."""
        from ner_controller.api.configs.text_router_config import TextRouterConfig

        config = TextRouterConfig()

        self.assertNotEqual(config, "/text")
        self.assertNotEqual(config, {"prefix": "/text"})


class TestTextRouterConfigFieldTypes(unittest.TestCase):
    """Tests TextRouterConfig field type constraints."""

    def test_prefix_field_is_string(self) -> None:
        """prefix field accepts string values."""
        from ner_controller.api.configs.text_router_config import TextRouterConfig

        config = TextRouterConfig(prefix="/text")

        self.assertIsInstance(config.prefix, str)

    def test_tags_field_is_tuple(self) -> None:
        """tags field accepts tuple values."""
        from ner_controller.api.configs.text_router_config import TextRouterConfig

        config = TextRouterConfig(tags=("tag1", "tag2"))

        self.assertIsInstance(config.tags, tuple)

    def test_tags_with_single_element(self) -> None:
        """tags field accepts single-element tuple."""
        from ner_controller.api.configs.text_router_config import TextRouterConfig

        config = TextRouterConfig(tags=("single-tag",))

        self.assertEqual(len(config.tags), 1)

    def test_tags_with_multiple_elements(self) -> None:
        """tags field accepts multiple-element tuple."""
        from ner_controller.api.configs.text_router_config import TextRouterConfig

        config = TextRouterConfig(tags=("tag1", "tag2", "tag3", "tag4"))

        self.assertEqual(len(config.tags), 4)

    def test_empty_prefix_is_allowed(self) -> None:
        """prefix field can be empty string (edge case)."""
        from ner_controller.api.configs.text_router_config import TextRouterConfig

        config = TextRouterConfig(prefix="")

        self.assertEqual(config.prefix, "")

    def test_empty_tags_tuple_is_allowed(self) -> None:
        """tags field can be empty tuple (edge case)."""
        from ner_controller.api.configs.text_router_config import TextRouterConfig

        config = TextRouterConfig(tags=())

        self.assertEqual(len(config.tags), 0)


class TestTextRouterConfigSpecialValues(unittest.TestCase):
    """Tests TextRouterConfig with special values."""

    def test_prefix_with_unicode(self) -> None:
        """prefix field accepts Unicode characters."""
        from ner_controller.api.configs.text_router_config import TextRouterConfig

        config = TextRouterConfig(prefix="/текст")

        self.assertEqual(config.prefix, "/текст")

    def test_tags_with_unicode(self) -> None:
        """tags field accepts Unicode strings."""
        from ner_controller.api.configs.text_router_config import TextRouterConfig

        config = TextRouterConfig(tags=("тег1", "тег2", "тег3"))

        self.assertIn("тег1", config.tags)

    def test_prefix_with_special_chars(self) -> None:
        """prefix field accepts special characters."""
        from ner_controller.api.configs.text_router_config import TextRouterConfig

        config = TextRouterConfig(prefix="/api/v1/text-processing")

        self.assertIn("/", config.prefix)
        self.assertIn("-", config.prefix)

    def test_tags_with_special_chars(self) -> None:
        """tags field accepts special characters."""
        from ner_controller.api.configs.text_router_config import TextRouterConfig

        config = TextRouterConfig(tags=("text-processing", "ner_api", "v1.0"))

        self.assertIn("text-processing", config.tags)


class TestTextRouterConfigStringRepresentation(unittest.TestCase):
    """Tests TextRouterConfig string representation."""

    def test_repr_contains_class_name(self) -> None:
        """String representation contains class name."""
        from ner_controller.api.configs.text_router_config import TextRouterConfig

        config = TextRouterConfig()

        repr_str = repr(config)
        self.assertIn("TextRouterConfig", repr_str)

    def test_repr_contains_field_values(self) -> None:
        """String representation contains field values."""
        from ner_controller.api.configs.text_router_config import TextRouterConfig

        config = TextRouterConfig(prefix="/custom", tags=("custom-tag",))

        repr_str = repr(config)
        self.assertIn("/custom", repr_str)
        self.assertIn("custom-tag", repr_str)
