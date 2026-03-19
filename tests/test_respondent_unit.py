import unittest
from unittest.mock import MagicMock, patch
import json
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.survey_agent import survey_node
from src.agents.persona_agent import persona_node
from src.agents.respondent_agent import respondent_node


class TestFullWorkflow(unittest.TestCase):
    """
    完整工作流集成测试。
    
    测试三个流程：
    1. 问卷生成
    2. 人格生成（6个）
    3. 人格答题
    """
    
    def setUp(self):
        self.initial_state = {
            "thread_id": "test_workflow_001",
            "topic": "企业内部用户中心模块的移动端适配体验调研",
            "questionnaire": {},
            "personas": [],
            "raw_data_path": "",
            "plot_image_paths": [],
            "analysis_insights": {},
            "thesis_draft": "",
            "error_logs": [],
            "current_step": "start",
            "persona_count": 6
        }
    
    @patch('src.agents.survey_agent.OpenAI')
    @patch('os.makedirs')
    @patch('builtins.open', create=True)
    def test_survey_generation(self, mock_openai, mock_makedirs, mock_open):
        """
        测试问卷生成节点。
        """
        print("\n" + "=" * 60)
        print("测试 1：问卷生成节点")
        print("=" * 60)
        
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = json.dumps({
            "survey_title": "用户中心模块移动端适配体验调研",
            "demographics": [
                {"id": "d1", "question": "您的年龄段", "options": ["18-25岁", "26-35岁", "36-45岁", "46-55岁", "55岁以上"]}
            ],
            "likert_scales": [
                {"id": "l1", "question": "移动端页面加载速度", "scale_range": [1, 5], "labels": {"1": "极慢", "5": "极快"}},
                {"id": "l2", "question": "移动端操作便捷性", "scale_range": [1, 5], "labels": {"1": "极差", "5": "极好"}},
                {"id": "l3", "question": "移动端界面美观度", "scale_range": [1, 5], "labels": {"1": "极差", "5": "极好"}},
                {"id": "l4", "question": "移动端功能完整性", "scale_range": [1, 5], "labels": {"1": "极差", "5": "极好"}},
                {"id": "l5", "question": "移动端稳定性", "scale_range": [1, 5], "labels": {"1": "极差", "5": "极好"}}
            ],
            "open_ended": [
                {"id": "o1", "question": "请描述您在使用移动端用户中心时遇到的最大痛点"},
                {"id": "o2", "question": "请描述一个您认为移动端用户中心最需要改进的功能"}
            ]
        })
        mock_client.chat.completions.create.return_value = mock_completion
        
        result = survey_node(self.initial_state)
        
        self.assertEqual(result["current_step"], "survey_agent")
        self.assertIn("questionnaire", result)
        self.assertEqual(result["questionnaire"]["survey_title"], "用户中心模块移动端适配体验调研")
        self.assertEqual(len(result["questionnaire"]["demographics"]), 1)
        self.assertEqual(len(result["questionnaire"]["likert_scales"]), 5)
        self.assertEqual(len(result["questionnaire"]["open_ended"]), 2)
        
        print("✅ 问卷生成测试通过")
        
        return result
    
    @patch('src.agents.persona_agent.OpenAI')
    @patch('os.makedirs')
    @patch('builtins.open', create=True)
    def test_persona_generation(self, mock_openai, mock_makedirs, mock_open):
        """
        测试人格生成节点（6个画像）。
        """
        print("\n" + "=" * 60)
        print("测试 2：人格生成节点（6个画像）")
        print("=" * 60)
        
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = json.dumps({
            "personas": [
                {
                    "name_tag": "后端开发-张伟",
                    "gender": "男",
                    "age": 32,
                    "job": "后端开发工程师",
                    "personality": "对系统性能极度敏感，追求极致效率，技术保守派，喜欢稳定可靠的方案",
                    "location": "北京海淀",
                    "proportion": 0.25,
                    "demographics_fixed": {"d1": "26-35岁"},
                    "likert_distribution": {
                        "l1": {"mu": 2.5, "sigma": 0.8},
                        "l2": {"mu": 3.0, "sigma": 0.7},
                        "l3": {"mu": 2.0, "sigma": 0.9},
                        "l4": {"mu": 2.5, "sigma": 0.6},
                        "l5": {"mu": 3.0, "sigma": 0.8}
                    },
                    "open_ended_samples": {
                        "o1": "每次打开用户中心都要等个三五秒，真的很烦人，特别是急着处理工单的时候",
                        "o2": "希望能有个快速入口，不用每次都点那么多次"
                    }
                },
                {
                    "name_tag": "产品经理-李娜",
                    "gender": "女",
                    "age": 28,
                    "job": "互联网产品经理",
                    "personality": "注重用户体验，对界面美观度要求极高，喜欢简洁直观的设计",
                    "location": "上海徐汇",
                    "proportion": 0.20,
                    "demographics_fixed": {"d1": "26-35岁"},
                    "likert_distribution": {
                        "l1": {"mu": 3.5, "sigma": 0.6},
                        "l2": {"mu": 4.0, "sigma": 0.5},
                        "l3": {"mu": 4.5, "sigma": 0.7},
                        "l4": {"mu": 4.0, "sigma": 0.6},
                        "l5": {"mu": 3.5, "sigma": 0.5}
                    },
                    "open_ended_samples": {
                        "o1": "界面有时候太复杂了，找不到想要的功能",
                        "o2": "希望能有更多个性化设置选项"
                    }
                },
                {
                    "name_tag": "运营专员-王强",
                    "gender": "男",
                    "age": 35,
                    "job": "用户运营专员",
                    "personality": "务实派，关注业务效率，对新功能接受度中等",
                    "location": "广州天河",
                    "proportion": 0.15,
                    "demographics_fixed": {"d1": "36-45岁"},
                    "likert_distribution": {
                        "l1": {"mu": 3.0, "sigma": 0.7},
                        "l2": {"mu": 3.5, "sigma": 0.6},
                        "l3": {"mu": 3.0, "sigma": 0.8},
                        "l4": {"mu": 3.5, "sigma": 0.7},
                        "l5": {"mu": 3.0, "sigma": 0.6}
                    },
                    "open_ended_samples": {
                        "o1": "有时候会卡顿，影响工作效率",
                        "o2": "希望能简化操作流程"
                    }
                },
                {
                    "name_tag": "测试工程师-刘敏",
                    "gender": "女",
                    "age": 29,
                    "job": "软件测试工程师",
                    "personality": "对系统稳定性要求极高，对bug容忍度低，喜欢详细的测试报告",
                    "location": "深圳南山",
                    "proportion": 0.15,
                    "demographics_fixed": {"d1": "26-35岁"},
                    "likert_distribution": {
                        "l1": {"mu": 2.0, "sigma": 0.9},
                        "l2": {"mu": 2.5, "sigma": 0.8},
                        "l3": {"mu": 2.0, "sigma": 0.7},
                        "l4": {"mu": 2.5, "sigma": 0.8},
                        "l5": {"mu": 2.0, "sigma": 0.9}
                    },
                    "open_ended_samples": {
                        "o1": "希望能有更多测试工具",
                        "o2": "希望能快速定位问题"
                    }
                },
                {
                    "name_tag": "前端开发-陈明",
                    "gender": "男",
                    "age": 31,
                    "job": "前端开发工程师",
                    "personality": "注重视觉效果和交互体验，对性能要求中等，喜欢创新的设计",
                    "location": "杭州西湖",
                    "proportion": 0.10,
                    "demographics_fixed": {"d1": "26-35岁"},
                    "likert_distribution": {
                        "l1": {"mu": 4.0, "sigma": 0.5},
                        "l2": {"mu": 4.5, "sigma": 0.6},
                        "l3": {"mu": 4.0, "sigma": 0.5},
                        "l4": {"mu": 4.5, "sigma": 0.6},
                        "l5": {"mu": 4.0, "sigma": 0.5}
                    },
                    "open_ended_samples": {
                        "o1": "动画效果可以更流畅一些",
                        "o2": "希望能支持更多设备"
                    }
                },
                {
                    "name_tag": "客服代表-赵芳",
                    "gender": "女",
                    "age": 27,
                    "job": "客户服务代表",
                    "personality": "耐心细致，关注用户满意度，对系统功能熟悉度中等",
                    "location": "成都武侯",
                    "proportion": 0.15,
                    "demographics_fixed": {"d1": "26-35岁"},
                    "likert_distribution": {
                        "l1": {"mu": 3.5, "sigma": 0.6},
                        "l2": {"mu": 3.0, "sigma": 0.7},
                        "l3": {"mu": 3.5, "sigma": 0.6},
                        "l4": {"mu": 3.0, "sigma": 0.7},
                        "l5": {"mu": 3.5, "sigma": 0.6}
                    },
                    "open_ended_samples": {
                        "o1": "希望能有更快捷的工单处理",
                        "o2": "希望能有知识库支持"
                    }
                }
            ]
        })
        mock_client.chat.completions.create.return_value = mock_completion
        
        state_with_questionnaire = {
            **self.initial_state,
            "questionnaire": {
                "survey_title": "用户中心模块移动端适配体验调研",
                "demographics": [
                    {"id": "d1", "question": "您的年龄段", "options": ["18-25岁", "26-35岁", "36-45岁", "46-55岁", "55岁以上"]}
                ],
                "likert_scales": [
                    {"id": "l1", "question": "移动端页面加载速度", "scale_range": [1, 5], "labels": {"1": "极慢", "5": "极快"}},
                    {"id": "l2", "question": "移动端操作便捷性", "scale_range": [1, 5], "labels": {"1": "极差", "5": "极好"}},
                    {"id": "l3", "question": "移动端界面美观度", "scale_range": [1, 5], "labels": {"1": "极差", "5": "极好"}},
                    {"id": "l4", "question": "移动端功能完整性", "scale_range": [1, 5], "labels": {"1": "极差", "5": "极好"}},
                    {"id": "l5", "question": "移动端稳定性", "scale_range": [1, 5], "labels": {"1": "极差", "5": "极好"}}
                ],
                "open_ended": [
                    {"id": "o1", "question": "请描述您在使用移动端用户中心时遇到的最大痛点"},
                    {"id": "o2", "question": "请描述一个您认为移动端用户中心最需要改进的功能"}
                ]
            }
        }
        
        result = persona_node(state_with_questionnaire)
        
        self.assertEqual(result["current_step"], "persona_agent")
        self.assertIn("personas", result)
        self.assertEqual(len(result["personas"]), 6)
        
        total_proportion = sum(p.get("proportion", 0) for p in result["personas"])
        self.assertAlmostEqual(total_proportion, 1.0, places=2)
        
        print("✅ 人格生成测试通过（6个画像）")
        
        return result
    
    @patch('src.agents.respondent_agent.OpenAI')
    @patch('os.makedirs')
    @patch('builtins.open', create=True)
    def test_respondent_generation(self, mock_openai, mock_makedirs, mock_open):
        """
        测试人格答题节点。
        """
        print("\n" + "=" * 60)
        print("测试 3：人格答题节点")
        print("=" * 60)
        
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = json.dumps({
            "responses": {
                "l1": 3,
                "l2": 4,
                "l3": 2,
                "l4": 3,
                "l5": 4,
                "o1": "每次打开用户中心都要等个三五秒，真的很烦人，特别是急着处理工单的时候",
                "o2": "希望能有个快速入口，不用每次都点那么多次",
                "d1": "26-35岁"
            }
        })
        mock_client.chat.completions.create.return_value = mock_completion
        
        state_with_personas = {
            **self.initial_state,
            "personas": [
                {
                    "name_tag": "后端开发-张伟",
                    "age": 32,
                    "job": "后端开发工程师",
                    "personality": "对系统性能极度敏感，追求极致效率，技术保守派，喜欢稳定可靠的方案"
                }
            ],
            "questionnaire": {
                "survey_title": "用户中心模块移动端适配体验调研",
                "demographics": [
                    {"id": "d1", "question": "您的年龄段", "options": ["18-25岁", "26-35岁", "36-45岁", "46-55岁", "55岁以上"]}
                ],
                "likert_scales": [
                    {"id": "l1", "question": "移动端页面加载速度", "scale_range": [1, 5], "labels": {"1": "极慢", "5": "极快"}},
                    {"id": "l2", "question": "移动端操作便捷性", "scale_range": [1, 5], "labels": {"1": "极差", "5": "极好"}},
                    {"id": "l3", "question": "移动端界面美观度", "scale_range": [1, 5], "labels": {"1": "极差", "5": "极好"}},
                    {"id": "l4", "question": "移动端功能完整性", "scale_range": [1, 5], "labels": {"1": "极差", "5": "极好"}},
                    {"id": "l5", "question": "移动端稳定性", "scale_range": [1, 5], "labels": {"1": "极差", "5": "极好"}}
                ],
                "open_ended": [
                    {"id": "o1", "question": "请描述您在使用移动端用户中心时遇到的最大痛点"},
                    {"id": "o2", "question": "请描述一个您认为移动端用户中心最需要改进的功能"}
                ]
            }
        }
        
        result = respondent_node(state_with_personas)
        
        self.assertEqual(result["current_step"], "respondent_agent")
        self.assertIn("seed_responses", result)
        self.assertEqual(len(result["seed_responses"]), 1)
        self.assertEqual(result["seed_responses"][0]["persona_name"], "后端开发-张伟")
        self.assertEqual(result["seed_responses"][0]["responses"]["l1"], 3)
        
        print("✅ 人格答题测试通过")
    
    @patch('src.agents.survey_agent.OpenAI')
    @patch('src.agents.persona_agent.OpenAI')
    @patch('src.agents.respondent_agent.OpenAI')
    @patch('os.makedirs')
    @patch('builtins.open', create=True)
    def test_full_workflow_integration(self, mock_openai, mock_makedirs, mock_open):
        """
        测试完整的三个流程集成。
        """
        print("\n" + "=" * 60)
        print("测试完整工作流集成")
        print("=" * 60)
        
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        
        state = self.initial_state
        
        print("\n步骤 1：问卷生成")
        mock_completion.choices[0].message.content = json.dumps({
            "survey_title": "用户中心模块移动端适配体验调研",
            "demographics": [
                {"id": "d1", "question": "您的年龄段", "options": ["18-25岁", "26-35岁", "36-45岁", "46-55岁", "55岁以上"]}
            ],
            "likert_scales": [
                {"id": "l1", "question": "移动端页面加载速度", "scale_range": [1, 5], "labels": {"1": "极慢", "5": "极快"}},
                {"id": "l2", "question": "移动端操作便捷性", "scale_range": [1, 5], "labels": {"1": "极差", "5": "极好"}},
                {"id": "l3", "question": "移动端界面美观度", "scale_range": [1, 5], "labels": {"1": "极差", "5": "极好"}},
                {"id": "l4", "question": "移动端功能完整性", "scale_range": [1, 5], "labels": {"1": "极差", "5": "极好"}},
                {"id": "l5", "question": "移动端稳定性", "scale_range": [1, 5], "labels": {"1": "极差", "5": "极好"}}
            ],
            "open_ended": [
                {"id": "o1", "question": "请描述您在使用移动端用户中心时遇到的最大痛点"},
                {"id": "o2", "question": "请描述一个您认为移动端用户中心最需要改进的功能"}
            ]
        })
        mock_client.chat.completions.create.return_value = mock_completion
        state = survey_node(state)
        
        print("\n步骤 2：人格生成（6个）")
        mock_completion.choices[0].message.content = json.dumps({
            "personas": [
                {
                    "name_tag": "后端开发-张伟",
                    "gender": "男",
                    "age": 32,
                    "job": "后端开发工程师",
                    "personality": "对系统性能极度敏感，追求极致效率，技术保守派，喜欢稳定可靠的方案",
                    "proportion": 0.25,
                    "demographics_fixed": {"d1": "26-35岁"},
                    "likert_distribution": {
                        "l1": {"mu": 2.5, "sigma": 0.8},
                        "l2": {"mu": 3.0, "sigma": 0.7},
                        "l3": {"mu": 2.0, "sigma": 0.9},
                        "l4": {"mu": 2.5, "sigma": 0.6},
                        "l5": {"mu": 3.0, "sigma": 0.8}
                    },
                    "open_ended_samples": {
                        "o1": "每次打开用户中心都要等个三五秒，真的很烦人，特别是急着处理工单的时候",
                        "o2": "希望能有个快速入口，不用每次都点那么多次"
                    }
                },
                {
                    "name_tag": "产品经理-李娜",
                    "gender": "女",
                    "age": 28,
                    "job": "互联网产品经理",
                    "personality": "注重用户体验，对界面美观度要求极高，喜欢简洁直观的设计",
                    "location": "上海徐汇",
                    "proportion": 0.20,
                    "demographics_fixed": {"d1": "26-35岁"},
                    "likert_distribution": {
                        "l1": {"mu": 3.5, "sigma": 0.6},
                        "l2": {"mu": 4.0, "sigma": 0.5},
                        "l3": {"mu": 4.5, "sigma": 0.7},
                        "l4": {"mu": 4.0, "sigma": 0.6},
                        "l5": {"mu": 4.5, "sigma": 0.5}
                    },
                    "open_ended_samples": {
                        "o1": "界面有时候太复杂了，找不到想要的功能",
                        "o2": "希望能有更多个性化设置选项"
                    }
                },
                {
                    "name_tag": "运营专员-王强",
                    "gender": "男",
                    "age": 35,
                    "job": "用户运营专员",
                    "personality": "务实派，关注业务效率，对新功能接受度中等",
                    "location": "广州天河",
                    "proportion": 0.15,
                    "demographics_fixed": {"d1": "36-45岁"},
                    "likert_distribution": {
                        "l1": {"mu": 3.0, "sigma": 0.7},
                        "l2": {"mu": 3.5, "sigma": 0.6},
                        "l3": {"mu": 3.0, "sigma": 0.8},
                        "l4": {"mu": 3.5, "sigma": 0.7},
                        "l5": {"mu": 3.0, "sigma": 0.6}
                    },
                    "open_ended_samples": {
                        "o1": "有时候会卡顿，影响工作效率",
                        "o2": "希望能简化操作流程"
                    }
                },
                {
                    "name_tag": "测试工程师-刘敏",
                    "gender": "女",
                    "age": 29,
                    "job": "软件测试工程师",
                    "personality": "对系统稳定性要求极高，对bug容忍度低，喜欢详细的测试报告",
                    "location": "深圳南山",
                    "proportion": 0.15,
                    "demographics_fixed": {"d1": "26-35岁"},
                    "likert_distribution": {
                        "l1": {"mu": 2.0, "sigma": 0.9},
                        "l2": {"mu": 2.5, "sigma": 0.8},
                        "l3": {"mu": 2.0, "sigma": 0.7},
                        "l4": {"mu": 2.5, "sigma": 0.8},
                        "l5": {"mu": 2.0, "sigma": 0.9}
                    },
                    "open_ended_samples": {
                        "o1": "希望能有更多测试工具",
                        "o2": "希望能快速定位问题"
                    }
                },
                {
                    "name_tag": "前端开发-陈明",
                    "gender": "男",
                    "age": 31,
                    "job": "前端开发工程师",
                    "personality": "注重视觉效果和交互体验，对性能要求中等，喜欢创新的设计",
                    "location": "杭州西湖",
                    "proportion": 0.10,
                    "demographics_fixed": {"d1": "26-35岁"},
                    "likert_distribution": {
                        "l1": {"mu": 4.0, "sigma": 0.5},
                        "l2": {"mu": 4.5, "sigma": 0.6},
                        "l3": {"mu": 4.0, "sigma": 0.5},
                        "l4": {"mu": 4.5, "sigma": 0.6},
                        "l5": {"mu": 4.0, "sigma": 0.5}
                    },
                    "open_ended_samples": {
                        "o1": "动画效果可以更流畅一些",
                        "o2": "希望能支持更多设备"
                    }
                },
                {
                    "name_tag": "客服代表-赵芳",
                    "gender": "女",
                    "age": 27,
                    "job": "客户服务代表",
                    "personality": "耐心细致，关注用户满意度，对系统功能熟悉度中等",
                    "location": "成都武侯",
                    "proportion": 0.15,
                    "demographics_fixed": {"d1": "26-35岁"},
                    "likert_distribution": {
                        "l1": {"mu": 3.5, "sigma": 0.6},
                        "l2": {"mu": 3.0, "sigma": 0.7},
                        "l3": {"mu": 3.5, "sigma": 0.6},
                        "l4": {"mu": 3.0, "sigma": 0.7},
                        "l5": {"mu": 3.5, "sigma": 0.6}
                    },
                    "open_ended_samples": {
                        "o1": "希望能有更快捷的工单处理",
                        "o2": "希望能有知识库支持"
                    }
                }
            ]
        })
        mock_client.chat.completions.create.return_value = mock_completion
        state = persona_node(state)
        
        print("\n步骤 3：人格答题")
        mock_completion.choices[0].message.content = json.dumps({
            "responses": {
                "l1": 3,
                "l2": 4,
                "l3": 2,
                "l4": 3,
                "l5": 4,
                "o1": "每次打开用户中心都要等个三五秒，真的很烦人，特别是急着处理工单的时候",
                "o2": "希望能有个快速入口，不用每次都点那么多次",
                "d1": "26-35岁"
            }
        })
        mock_client.chat.completions.create.return_value = mock_completion
        state = respondent_node(state)
        
        self.assertEqual(state["current_step"], "respondent_agent")
        self.assertIn("seed_responses", state)
        self.assertEqual(len(state["seed_responses"]), 1)
        
        print("\n" + "=" * 60)
        print("✅ 完整工作流集成测试通过")
        print("=" * 60)


if __name__ == '__main__':
    unittest.main()
