from collections import OrderedDict


def prettyformat_dict_string(d, indent=''):
    result = list()
    for k, v in d.items():
        if isinstance(v, dict):
            result.append('{}{}:\t\n{}'.format(indent, k, prettyformat_dict_string(v, indent + '  ')))
        else:
            result.append('{}{}:\t{}\n'.format(indent, k, v))
    return ''.join(result)


def capitalize(text: str) -> str:
    return text[0].upper() + text[1:]


def snip_anchor(text: str) -> str:
    snip_anchor = text.find('#')
    if snip_anchor != -1:
        text = text[:snip_anchor]
    return text


def normalize_wiki_entity(lst):
    memo = set()
    result = list()
    for i in lst:
        i = snip_anchor(i)
        if len(i) == 0: continue
        i = capitalize(i)
        if i not in memo:
            result.append(i)
            memo.add(i)
    return result

def snip(string, search, keep, keep_search):
    pos = string.find(search)
    if pos != -1:
        if keep == 'left':
            if keep_search:
                pos += len(search)
            string = string[:pos]
        if keep == 'right':
            if not keep_search:
                pos += len(search)
            string = string[pos:]
    return string


def merge(list_a, list_b, pr=False):
    result = OrderedDict()
    for n, c in list_a:
        result[n] = c
    for n, c in list_b:
        if n in result:
            result[n] = result[n] + c
        else:
            result[n] = c
    return sorted(result.items(), key=lambda x: x[1], reverse=True)


def redirect_entity(ent, redirects_en):
    if ent is not None:
        ent_underscore = ent.replace(' ', '_')
        if ent_underscore in redirects_en:
            ent = redirects_en[ent_underscore].replace('_', ' ')
    return ent
