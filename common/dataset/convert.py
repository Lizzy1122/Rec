import pandas as pd
import os


# 假设statics.csv的结构是：student_id,question_id,correct
# 其中correct为1表示正确，0表示错误


class Statics2011Converter:
    def __init__(self, statics_path, output_path):
        self.statics_path = statics_path
        self.output_path = output_path
        self.student_ids = []
        self.question_ids = set()

    def parse_statics_file(self, file_name):
        # 初始化一个空的列表 interactions，用于存储交互数据
        interactions = []
        with open(f'{self.statics_path}/{file_name}', 'r') as file:
            lines = file.readlines()

        for i in range(0, len(lines), 3):
            # 对于每组数据，第一行包含问题的数量（未在代码中使用），第二行是问题ID的列表，第三行是相应的正确答案标记（0或1）的列表。
            num_questions = int(lines[i].strip())
            questions = list(map(int, lines[i + 1].strip().split(',')))
            correct_answers = list(map(int, lines[i + 2].strip().split(',')))

            # 为每组数据动态生成一个新的学生ID，并将其添加到 self.student_ids 列表。
            student_id = len(self.student_ids)
            self.student_ids.append(student_id)

            for question_id, correct in zip(questions, correct_answers):
                self.question_ids.add(question_id)
                interactions.append((student_id, question_id, correct))
        return interactions

    def write_interactions_to_file(self, interactions, file_name):
        # 使用 with 语句和 open 函数打开（或创建）一个文件，用于写入数据。文件的路径是 self.output_path 和 file_name 的组合，模式 'w' 表示写入模式，如果文件已存在则会被覆盖。
        with open(f'{self.output_path}/{file_name}', 'w') as file:
            for interaction in interactions:
                # 将三元组的整数值写入文件，元素之间用空格分隔
                file.write(f'{interaction[0]} {interaction[1]} {interaction[2]}\n')

    def generate_entity_item_user_lists(self):
        with open(f'{self.output_path}/entity_list.dat', 'w') as f_ent, \
                open(f'{self.output_path}/item_list.dat', 'w') as f_item, \
                open(f'{self.output_path}/user_list.dat', 'w') as f_user:
            for student_id in self.student_ids:
                f_ent.write(f'{student_id}\n')
                f_user.write(f'{student_id}\n')
            for question_id in self.question_ids:
                f_ent.write(f'{question_id}\n')
                f_item.write(f'{question_id}\n')

    def convert(self):
        # Parse each file and generate interactions
        for i in range(1, 6):
            train_interactions = self.parse_statics_file(f'statics_train{i}.csv')
            self.write_interactions_to_file(train_interactions, f'train{i}.dat')
            test_interactions = self.parse_statics_file(f'statics_test{i}.csv')
            self.write_interactions_to_file(test_interactions, f'test{i}.dat')
            valid_interactions = self.parse_statics_file(f'statics_valid{i}.csv')
            self.write_interactions_to_file(valid_interactions, f'valid{i}.dat')

        # Generate entity, item, and user lists
        self.generate_entity_item_user_lists()

# Usage example


converter = Statics2011Converter('D:\\Developer\\Python_workspace\\KG-Policy-test\\Data\\statics',
                                 'D:\\Developer\\Python_workspace\\KG-Policy-test\\Data\\mydata')
converter.convert()
