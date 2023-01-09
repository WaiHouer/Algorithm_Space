"""
列表-嵌套字典，按照字典键值进行去重
"""


def duplicate_removal(datas, condition, model="key"):
    """
    :param datas: 准备去重的数据，格式[{},{}...]
    :param condition: 去重参考的键值，需要数据里面有这些key
    :param model: 去重模式，key模式为去重参考的key值；notkey模式为去重不参考的key值。相反关系。
    :return: 去重后的数据，格式[{},{}...]
    """
    def flags(keys, data):
        tmp_dic = {}
        for key in keys:
            tmp_dic.update({key: data.get(key)})
        return tmp_dic

    removal_data = []
    values = []
    if datas:
        if model == "key":
            keys = condition
        elif model == "notkey":
            keys = [key for key in datas[0].keys() if key not in condition]
        else:
            raise ValueError("传入的model值错误，无法匹配")
        for data in datas:
            if flags(keys, data) not in values:
                removal_data.append(data)
                values.append(flags(keys, data))

    return removal_data