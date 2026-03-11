from typing import Generator

from dify_plugin.entities.model import ModelFeature
from dify_plugin.entities.model.message import PromptMessageContentType, PromptMessage
from dify_plugin.interfaces.agent import AgentModelConfig


MODEL_FEATURES_MAPPING = {
    PromptMessageContentType.IMAGE: {ModelFeature.VISION, },
    PromptMessageContentType.VIDEO: {ModelFeature.VISION, ModelFeature.VIDEO, },
    PromptMessageContentType.DOCUMENT: {ModelFeature.VISION, ModelFeature.DOCUMENT, },
    PromptMessageContentType.AUDIO: {ModelFeature.AUDIO, },
}


class FilterHistoryMessageByModelFeaturesMixin:

    @staticmethod
    def _iter_cleanup_history_prompt_messages(model: AgentModelConfig) -> Generator[PromptMessage, None, None]:
        """
        remove history_prompt_message if model not support
        :param model
        :return:
        """
        model_features = set(model.entity.features)
        for msg in model.history_prompt_messages:
            if isinstance(msg.content, list):
                filtered_content = [
                    item
                    for item in msg.content
                    if (
                            item.type == PromptMessageContentType.TEXT
                            or bool(MODEL_FEATURES_MAPPING[item.type] & model_features)
                    )
                ]
                new_msg = msg.__class__(
                    role=msg.role,
                    content=filtered_content,
                    name=msg.name,
                )
                yield new_msg
            else:
                yield msg
