# pii_masker_br.py
import json
import logging
import operator
import re
import sys
import unicodedata
from collections.abc import Iterable
from typing import Optional

# ============================== logging ==============================
LOGGER = logging.getLogger('pii_masker')
if not LOGGER.handlers:
    h = logging.StreamHandler(sys.stderr)
    fmt = logging.Formatter('[%(levelname)s] %(message)s')
    h.setFormatter(fmt)
    LOGGER.addHandler(h)
LOGGER.setLevel(logging.INFO)


# ============================== helpers ==============================
def _strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def _digits_only(s: str) -> str:
    return re.sub(r'\D+', '', s or '')


# Replace curly quotes/dashes with ASCII; leave everything else intact
_REPLACE_CHARS = {'\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"', '\u2013': '-', '\u2014': '-'}


def _normalize_with_map(s: str) -> tuple[str, list[int]]:
    """Build a normalized string (lowercase, accentless, whitespace-collapsed) plus
    a map from normalized indices back to original indices in `s`.
    """
    out_chars: list[str] = []
    idx_map: list[int] = []
    last_w = False

    for i, ch in enumerate(s):
        ch = _REPLACE_CHARS.get(ch, ch)
        # NFD, drop combining marks
        decomp = unicodedata.normalize('NFD', ch)
        base = ''.join(c for c in decomp if unicodedata.category(c) != 'Mn')
        for bc in base:
            if bc.isspace():
                if last_w:
                    continue
                out_chars.append(' ')
                idx_map.append(i)
                last_w = True
            else:
                out_chars.append(bc.lower())
                idx_map.append(i)
                last_w = False
    return ''.join(out_chars), idx_map


def _normalize_simple(s: str) -> str:
    ns, _ = _normalize_with_map(s)
    return ns


# Brazilian UFs
UF_SET = {
    'AC',
    'AL',
    'AP',
    'AM',
    'BA',
    'CE',
    'DF',
    'ES',
    'GO',
    'MA',
    'MT',
    'MS',
    'MG',
    'PA',
    'PB',
    'PR',
    'PE',
    'PI',
    'RJ',
    'RN',
    'RS',
    'RO',
    'RR',
    'SC',
    'SP',
    'SE',
    'TO',
}
UF_NAME_TO_CODE = {
    'ACRE': 'AC',
    'ALAGOAS': 'AL',
    'AMAPÁ': 'AP',
    'AMAPA': 'AP',
    'AMAZONAS': 'AM',
    'BAHIA': 'BA',
    'CEARÁ': 'CE',
    'CEARA': 'CE',
    'DISTRITO FEDERAL': 'DF',
    'ESPÍRITO SANTO': 'ES',
    'ESPIRITO SANTO': 'ES',
    'GOIÁS': 'GO',
    'GOIAS': 'GO',
    'MARANHÃO': 'MA',
    'MARANHAO': 'MA',
    'MATO GROSSO': 'MT',
    'MATO GROSSO DO SUL': 'MS',
    'MINAS GERAIS': 'MG',
    'PARÁ': 'PA',
    'PARA': 'PA',
    'PARAÍBA': 'PB',
    'PARAIBA': 'PB',
    'PARANÁ': 'PR',
    'PARANA': 'PR',
    'PERNAMBUCO': 'PE',
    'PIAUÍ': 'PI',
    'PIAUI': 'PI',
    'RIO DE JANEIRO': 'RJ',
    'RIO GRANDE DO NORTE': 'RN',
    'RIO GRANDE DO SUL': 'RS',
    'RONDÔNIA': 'RO',
    'RONDONIA': 'RO',
    'RORAIMA': 'RR',
    'SANTA CATARINA': 'SC',
    'SÃO PAULO': 'SP',
    'SAO PAULO': 'SP',
    'SERGIPE': 'SE',
    'TOCANTINS': 'TO',
}

# RG authorities (broad coverage of common abbreviations)
RG_AUTHORITIES = [
    'SSP',
    'SESP',
    'DETRAN',
    'IFP',
    'PC',
    'DGPC',
    'SPTC',
    'SEGUP',
    'SDS',
    'POLICIA CIVIL',
    'POLÍCIA CIVIL',
    'POLICIA-CIVIL',
    'POLÍCIA-CIVIL',
]

# Accent-flexible character classes for recall on names & addresses
VOWEL_CLASSES = {
    'a': '[aàáâã]',
    'e': '[eèéê]',
    'i': '[iìíî]',
    'o': '[oòóôõ]',
    'u': '[uùúû]',
    'c': '[cç]',
    'A': '[AÀÁÂÃ]',
    'E': '[EÈÉÊ]',
    'I': '[IÌÍÎ]',
    'O': '[OÒÓÔÕ]',
    'U': '[UÙÚÛ]',
    'C': '[CÇ]',
}


def accent_flexible_regex(text: str) -> str:
    out = [VOWEL_CLASSES.get(ch, re.escape(ch)) for ch in text]
    return ''.join(out)


# ============================== priorities ==============================
# NOTE: We do not invent new types. This map just defines ordering when conflicts happen.
PRIORITY = {
    'EMAIL': 100,
    'URL': 96,
    'CPF': 95,
    'CNPJ': 94,
    'RG': 93,
    'PHONE': 92,
    'ZIPCODE': 91,
    'CEP': 91,  # CEP honored explicitly
    'BIRTHDATE': 91,
    # narrative/sensitive
    'RELIGIOUS_CONVICTION': 90,
    'POLITICAL_OPINION': 90,
    'RACE_OR_ETHNICITY': 90,
    'ORGANIZATION_AFFILIATION': 90,
    'MEDICAL_DATA': 90,
    'SEXUAL_DATA': 90,
    'MEDICAL_CONDITION': 90,
    # location/name
    'STREET': 80,
    'NEIGHBORHOOD': 80,
    'CITY_UF': 79,
    'BUILDING_NUMBER': 78,
    'SURNAMES': 60,
    'MIDDLE_NAME': 60,
    'NAME': 55,
    'AGE': 1,
    'DEFAULT': 50,
}


def _pii_priority(t: str) -> int:
    return PRIORITY.get(t.upper(), PRIORITY['DEFAULT'])


SENSITIVE_FUZZY = {
    'RACE_OR_ETHNICITY',
    'RELIGIOUS_CONVICTION',
    'POLITICAL_OPINION',
    'ORGANIZATION_AFFILIATION',
    'MEDICAL_DATA',
    'SEXUAL_DATA',
    'MEDICAL_CONDITION',
}
SENSITIVE_PAREN = {
    'RACE_OR_ETHNICITY',
    'RELIGIOUS_CONVICTION',
    'POLITICAL_OPINION',
    'ORGANIZATION_AFFILIATION',
    'MEDICAL_DATA',
    'SEXUAL_DATA',
    'MEDICAL_CONDITION',
}


# ============================== parentheses handling ==============================
def _wrap_parentheses_span(text: str, start: int, end: int):
    # Expand to include wrapping (...) if they tightly wrap the match (allow whitespace inside).
    l = start - 1
    while l >= 0 and text[l].isspace():
        l -= 1
    r = end
    while r < len(text) and text[r].isspace():
        r += 1
    if l >= 0 and text[l] == '(' and r < len(text) and text[r] == ')':
        return l, r + 1, True
    return start, end, False


# ============================== type-aware patterns (exact types) ==============================
def _pat_email(email: str) -> re.Pattern:
    if '@' not in email:
        return re.compile(re.escape(email), re.IGNORECASE)
    local, dom = email.split('@', 1)
    return re.compile(rf'(?i)(?<![A-Za-z0-9._%+\-]){re.escape(local)}@{re.escape(dom)}(?=$|[\s,;:)\]\.!\?])')


def _pat_cpf(cpf: str) -> re.Pattern:
    d = _digits_only(cpf)
    if len(d) != 11:
        return re.compile(re.escape(cpf), re.IGNORECASE)
    dotted = rf'{d[:3]}\.{d[3:6]}\.{d[6:9]}-{d[9:11]}'
    return re.compile(rf'(?<!\d)(?:{d}|{dotted})(?!\d)')


def _pat_zipcode(cep: str) -> re.Pattern:
    d = _digits_only(cep)
    if len(d) == 8:
        dotted = f'{d[:5]}-{d[5:]}'
        return re.compile(rf'(?<!\d)(?:{d}|{re.escape(dotted)})(?!\d)')
    return re.compile(re.escape(cep), re.IGNORECASE)


def _pat_rg_digits_flexible(digits: str) -> str:
    # same digits in order; allow optional separators between digits
    return ''.join(re.escape(ch) + r'(?:[ .-]?)' for ch in digits)


def _pat_rg(rg: str) -> re.Pattern:
    d = _digits_only(rg)
    if not d:
        return re.compile(re.escape(rg), re.IGNORECASE)
    num = _pat_rg_digits_flexible(d)
    auths = '|'.join(sorted({re.escape(a) for a in RG_AUTHORITIES}, key=len, reverse=True))
    uf = r'(?:AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MT|MS|MG|PA|PB|PR|PE|PI|RJ|RN|RS|RO|RR|SC|SP|SE|TO)'
    auth_block = rf'(?:[,.\s]+(?:emitido pel[ao]\s+)?(?:{auths})(?:[ \-\/]*{uf})?)?'
    return re.compile(rf'(?i)(?<!\d){num}(?!\d){auth_block}')


def _pat_phone(phone: str) -> re.Pattern:
    d = _digits_only(phone)
    if len(d) < 8:
        return re.compile(re.escape(phone), re.IGNORECASE)
    pre = r'(?:\+?55[ .-]*)?'  # optional +55
    if len(d) in {10, 11}:
        ac = d[:2]
        rest = d[2:]
        rest_core = ''.join(re.escape(ch) + r'(?:[ .-]*)' for ch in rest).rstrip('(?:[ .-]*)')
        core = rf'(?:\({ac}\)|{ac})(?:[ .-]*){rest_core}'
        return re.compile(pre + core)
    core = ''.join(re.escape(ch) + r'(?:[ .-]*)' for ch in d).rstrip('(?:[ .-]*)')
    return re.compile(pre + core)


def _pat_building_number(value: str) -> re.Pattern:
    v = value.strip().lower()
    if v in {'s/n', 'sem numero', 'sem número'}:
        return re.compile(r'(?i)\b(?:s/n|sem número|sem numero)\b')
    return re.compile(rf'(?i)\b{re.escape(value)}\b')


def _pat_name_like(value: str, accent_flexible: bool = True) -> re.Pattern:
    if accent_flexible:
        core = accent_flexible_regex(value)
        return re.compile(rf'(?i)(?<![@A-Za-z0-9._%+\-]){core}(?=$|[\s,;:)\]\.!\?])')
    return re.compile(rf'(?i)(?<![@A-Za-z0-9._%+\-]){re.escape(value)}(?=$|[\s,;:)\]\.!\?])')


def extend_to_word_boundary(text: str, s: int, e: int) -> tuple[int, int]:
    """Extend span to word boundaries if cutting off words."""
    # Extend start backward if we're mid-word
    while s > 0 and text[s - 1].isalnum():
        s -= 1
    # Extend end forward if we're mid-word
    while e < len(text) and text[e].isalnum():
        e += 1
    return s, e


def _pat_birthdate(value: str) -> re.Pattern:
    v = value.strip()
    m_iso = re.match(r'^(\d{4})[-/\.](\d{2})[-/\.](\d{2})$', v)
    m_br = re.match(r'^(\d{2})[-/\.](\d{2})[-/\.](\d{4})$', v)
    variants = set()
    if m_iso:
        y, mo, d = m_iso.groups()
        variants |= {f'{d}/{mo}/{y}', f'{d}-{mo}-{y}', f'{d}.{mo}.{y}', f'{y}-{mo}-{d}', f'{y}/{mo}/{d}'}
    elif m_br:
        d, mo, y = m_br.groups()
        variants |= {f'{d}/{mo}/{y}', f'{d}-{mo}-{y}', f'{d}.{mo}.{y}', f'{y}-{mo}-{d}', f'{y}/{mo}/{d}'}
    else:
        variants.add(v)
    pats = [re.escape(x) for x in variants]
    return re.compile(rf'(?<!\d)(?:{"|".join(pats)})(?!\d)', re.IGNORECASE)


def _pat_city_uf(value: str) -> re.Pattern:
    # Extract UF(s) from provided value (by code or full name). Mask only the UF token(s).
    val_up = _strip_accents(value).upper()
    ufs = {tok for tok in re.findall(r'\b[A-Z]{2}\b', val_up) if tok in UF_SET}
    if not ufs:
        for name, code in UF_NAME_TO_CODE.items():
            if name in val_up:
                ufs.add(code)
    if not ufs:
        # Fallback: literal (accent-flex) for the whole provided string
        return _pat_name_like(value, accent_flexible=True)
    uf_alt = '|'.join(sorted(ufs))
    return re.compile(rf'(?i)\b(?:{uf_alt})\b')


def _pattern_for_type(t: str, v: str) -> re.Pattern:
    """Do not invent types. We just accept the input key `t` (any case).
    For certain known aliases (CEP, TELEFONE, etc.) we apply type-appropriate patterns
    but we DO NOT change the label used for masking.
    """
    tu = t.upper()
    if tu == 'EMAIL':
        return _pat_email(v)
    if tu == 'CPF':
        return _pat_cpf(v)
    if tu == 'RG':
        return _pat_rg(v)
    if tu in {'PHONE', 'CEL', 'CELULAR', 'TELEFONE', 'FONE'}:
        return _pat_phone(v)
    if tu == 'BUILDING_NUMBER':
        return _pat_building_number(v)
    if tu in {'ZIPCODE', 'CEP', 'POSTAL_CODE', 'POSTCODE'}:
        return _pat_zipcode(v)
    if tu == 'CITY_UF':
        return _pat_city_uf(v)
    if tu == 'BIRTHDATE':
        return _pat_birthdate(v)
    if tu in {'NAME', 'SURNAMES', 'MIDDLE_NAME', 'NEIGHBORHOOD', 'STREET'}:
        return _pat_name_like(v, accent_flexible=True)
    if tu == 'AGE':
        try:
            n = int(str(v).strip())
            return re.compile(rf'(?<!\d){n}(?!\d)')
        except:
            return re.compile(re.escape(str(v)), re.IGNORECASE)
    return re.compile(re.escape(v), re.IGNORECASE)


# ============================== canonicalization (keys only) ==============================
def _canonicalize_types(pii: dict[str, str]) -> dict[str, str]:
    """We DO NOT introduce new mask types beyond items_used keys.
    We only normalize case and unify a few aliases (e.g., BAIRRO->NEIGHBORHOOD), still preserving the same concept.
    The output keys are UPPERCASE versions of the input keys (or their very close aliases).
    """
    out = {}
    for k, v in (pii or {}).items():
        if v is None:
            continue
        kk = k.strip().upper()
        # very close aliasing; still not inventing new semantics
        if kk in {'SURNAME', 'LAST_NAME', 'LASTNAME'}:
            kk = 'SURNAMES'
        if kk in {'HOOD', 'BAIRRO'}:
            kk = 'NEIGHBORHOOD'
        if kk in {'CITY-UF', 'CITYUF', 'UF_CITY', 'CITY/UF'}:
            kk = 'CITY_UF'
        if kk in {'NÚMERO', 'NUMERO', 'NUM', 'NUMERO_RESIDENCIA', 'NÚMERO_RESIDÊNCIA'}:
            kk = 'BUILDING_NUMBER'
        # CEP stays CEP; PHONE aliases handled in _pattern_for_type
        out[kk] = v
    return out


# ============================== fuzzy matching for narrative PII ==============================
def _fuzzy_substring_spans(text: str, value: str, min_chars: int = 6) -> list[tuple[int, int]]:
    """Progressive edge trimming: try value[l:r] for increasing total trim (from both sides),
    under a simple normalization (lowercase, accentless, whitespace-collapsed).
    Returns all non-overlapping occurrences (by normalized coordinates, mapped back to original).
    We lower min_chars to 5/4/3 if nothing matches, to "mask something".
    """
    if not value or not text:
        return []
    t_norm, t_map = _normalize_with_map(text)
    v_norm = _normalize_simple(value)
    n = len(v_norm)
    if n == 0 or len(t_norm) == 0:
        return []

    def find_first_cand(min_c: int) -> Optional[str]:
        for total_trim in range(n):  # total characters removed across both sides
            length = n - total_trim
            if length < min_c:
                break
            # Enumerate all splits of total_trim into left/right: l from 0..total_trim
            for l in range(total_trim + 1):
                r = n - (total_trim - l)
                cand = v_norm[l:r]
                idx = t_norm.find(cand)
                if idx != -1:
                    return cand
        return None

    for m in (min_chars, 5, 4, 3):
        cand = find_first_cand(m)
        if cand:
            # Get all occurrences of cand (non-overlapping) in normalized text
            spans: list[tuple[int, int]] = []
            start = 0
            L = len(cand)
            while True:
                pos = t_norm.find(cand, start)
                if pos == -1:
                    break
                s_orig = t_map[pos]
                e_orig = t_map[pos + L - 1] + 1
                spans.append((s_orig, e_orig))
                start = pos + L
            return spans
    return []


# ============================== masking core ==============================
def _is_safe_for_spacing(pos: int, masked_regions: list[tuple[int, int]]) -> bool:
    """Check if position is safe for adding spaces (not within or adjacent to existing masks)"""
    if pos <= 0:
        return False

    # Check if position conflicts with any existing masked region
    for start, end in masked_regions:
        # Don't add space if it would be inside a mask or immediately adjacent
        if start <= pos <= end or abs(pos - start) <= 1 or abs(pos - end) <= 1:
            return False

    return True


def create_masked_text(text: str, pii_values: dict[str, str], debug: bool = False) -> str:
    """Flexible-precision masker:
    - Only uses keys present in `pii_values` (after very close alias normalization).
    - Type-aware candidate spans for structured PII.
    - Fuzzy edge-trim for narrative PII (see SENSITIVE_FUZZY).
    - Resolves overlaps by priority (EMAIL > CPF > RG > ... > NAME).
    - Parentheses policy (keep for SENSITIVE_FUZZY; phone absorbs; others drop).
    - AGE > 80 => replace by literal '80'.
    """
    if not text:
        return ''

    pii = _canonicalize_types(pii_values)

    candidates: list[tuple[int, int, str, dict]] = []

    for t, v in pii.items():
        tu = t.upper()
        if tu in SENSITIVE_FUZZY:
            spans = _fuzzy_substring_spans(text, v, min_chars=6)
            if not spans and debug:
                LOGGER.debug(f'[FUZZY MISS] type={tu} len={len(v)} value_head={v[:40]!r}')
            for s, e in spans:
                candidates.append((s, e, t, {'value': v, 'fuzzy': True}))
        else:
            pat = _pattern_for_type(t, v)
            candidates.extend((m.start(), m.end(), t, {'value': v, 'fuzzy': False}) for m in pat.finditer(text))

    if not candidates:
        return text

    # Resolve overlaps by (priority, length)
    candidates.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    selected: list[tuple[int, int, str, dict]] = []

    def overlaps(a, b) -> bool:
        return not (a[1] <= b[0] or b[1] <= a[0])

    for c in candidates:
        take = True
        for s in list(selected):
            if overlaps(c, s):
                cs = (_pii_priority(c[2]), c[1] - c[0])
                ss = (_pii_priority(s[2]), s[1] - s[0])
                if cs > ss:
                    selected.remove(s)
                    break
                take = False
                break
        if take:
            selected.append(c)

    # Greedy clean to non-overlapping set
    selected.sort(key=lambda x: (_pii_priority(x[2]), (x[1] - x[0])), reverse=True)
    cleaned: list[tuple[int, int, str, dict]] = []
    for s, e, t, meta in selected:
        if all(e <= cs or s >= ce for cs, ce, _, _ in cleaned):
            cleaned.append((s, e, t, meta))

    # Replace from right to left, applying parentheses + age policies
    cleaned.sort(key=operator.itemgetter(0), reverse=True)
    out = text
    masked_regions: list[tuple[int, int]] = []  # Track already masked regions

    for s, e, t, meta in cleaned:
        tu = t.upper()
        rep = f'[{tu}]'
        if tu == 'AGE':
            try:
                n = int(str(meta.get('value')).strip())
                rep = '80' if n > 80 else '[AGE]'
            except:
                rep = '[AGE]'

        s2, e2, wrapped = _wrap_parentheses_span(out, s, e)
        if wrapped:
            if tu in SENSITIVE_PAREN:
                # keep parentheses visible around the mask
                out = out[:s2] + '(' + rep + ')' + out[e2:]
                masked_regions.append((s2, s2 + len('(' + rep + ')')))
                continue
            if tu in {'PHONE', 'CEL', 'CELULAR', 'TELEFONE', 'FONE'}:
                # absorb parentheses into the mask
                replacement = f'[{tu}]'
                out = out[:s2] + replacement + out[e2:]
                masked_regions.append((s2, s2 + len(replacement)))
                continue
            # drop parentheses entirely
            out = out[:s2] + rep + out[e2:]
            masked_regions.append((s2, s2 + len(rep)))
            continue

        # Add minimal spacing around replacement if needed - but avoid corrupting existing masks
        needs_space_before = (
            s > 0 and out[s - 1].isalnum() and not out[s - 1].isspace() and
            out[s - 1] not in '.,;:!?()[]{}"\'-/\\' and
            _is_safe_for_spacing(s, masked_regions)
        )
        needs_space_after = (
            e < len(out) and out[e].isalnum() and not out[e].isspace() and
            out[e] not in '.,;:!?()[]{}"\'-/\\' and
            _is_safe_for_spacing(e, masked_regions)
        )

        if needs_space_before:
            rep = ' ' + rep
        if needs_space_after:
            rep += ' '

        out = out[:s] + rep + out[e:]
        # Track this masked region (adjusting for any added spaces)
        start_pos = s - (1 if needs_space_before else 0)
        masked_regions.append((start_pos, start_pos + len(rep)))
    return out


# ============================== JSONL streaming driver ==============================
def _merge_items_used(items_used: Iterable[dict[str, str]]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for d in items_used or []:
        for k, v in (d or {}).items():
            if v is not None and v != '':
                merged[k] = v
    return merged

def process_jsonl_first_1000(input_path: str, output_path: str, text_field: str = 'generated_sample_PII') -> None:
        """Stream the first 1000 lines of a massive JSONL. For each line:
          - read `text_field` as input text
          - read `items_used` (list[dict]) and merge to pii dict
          - create masked text
          - write a compact JSONL with _id, masked, original length, keys, and (optionally) debug knobs
        Logs progress for every record and exceptions where relevant.
        """
        count = 0
        with open(input_path, encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
            for line_num, line in enumerate(fin, 1):
                if count >= 1000:
                    LOGGER.info('Reached 1000 records; stopping as requested.')
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception as e:
                    LOGGER.exception(f'JSON parse error at line {line_num}: {e}')
                    continue

                text = rec.get(text_field) or rec.get('text') or ''
                items_used = rec.get('items_used') or []
                pii = _merge_items_used(items_used)
                # After creating the masked text (line 432)
                masked = create_masked_text(text, pii, debug=False)

                # Add this new code to calculate matched/unmatched
                items_used_keys = set()
                for item_dict in items_used:
                    if isinstance(item_dict, dict):
                        items_used_keys.update(k.upper() for k in item_dict)

                pii_keys_set = {k.upper() for k in pii}
                matched = sorted(items_used_keys & pii_keys_set)
                unmatched = sorted(items_used_keys - pii_keys_set)

                # Modified output object
                out_obj = {
                    '_id': rec.get('_id'),
                    'masked': masked,
                    'original_len': len(text),
                    'pii_keys': sorted([k.upper() for k in pii]),
                    'measure': {'matched': matched, 'unmatched': unmatched},
                    'original_record': rec,  # if you kept this from previous change
                }

                fout.write(json.dumps(out_obj, ensure_ascii=False) + '\n')

                count += 1
                LOGGER.info(f'#{count} _id={rec.get("_id")} text_len={len(text)} pii={sorted(pii.keys())}')



def process_jsonl_first_100(input_path: str, output_path: str, text_field: str = 'generated_sample_PII') -> None:
    """Stream the first 100 lines of a massive JSONL. For each line:
      - read `text_field` as input text
      - read `items_used` (list[dict]) and merge to pii dict
      - create masked text
      - write a compact JSONL with _id, masked, original length, keys, and (optionally) debug knobs
    Logs progress for every record and exceptions where relevant.
    """
    count = 0
    with open(input_path, encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
        for line_num, line in enumerate(fin, 1):
            if count >= 100:
                LOGGER.info('Reached 100 records; stopping as requested.')
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as e:
                LOGGER.exception(f'JSON parse error at line {line_num}: {e}')
                continue

            text = rec.get(text_field) or rec.get('text') or ''
            items_used = rec.get('items_used') or []
            pii = _merge_items_used(items_used)
            # After creating the masked text (line 432)
            masked = create_masked_text(text, pii, debug=False)

            # Add this new code to calculate matched/unmatched
            items_used_keys = set()
            for item_dict in items_used:
                if isinstance(item_dict, dict):
                    items_used_keys.update(k.upper() for k in item_dict)

            pii_keys_set = {k.upper() for k in pii}
            matched = sorted(items_used_keys & pii_keys_set)
            unmatched = sorted(items_used_keys - pii_keys_set)

            # Modified output object
            out_obj = {
                '_id': rec.get('_id'),
                'masked': masked,
                'original_len': len(text),
                'pii_keys': sorted([k.upper() for k in pii]),
                'measure': {'matched': matched, 'unmatched': unmatched},
                'original_record': rec,  # if you kept this from previous change
            }

            fout.write(json.dumps(out_obj, ensure_ascii=False) + '\n')

            count += 1
            LOGGER.info(f'#{count} _id={rec.get("_id")} text_len={len(text)} pii={sorted(pii.keys())}')


# ============================== quick demos/tests ==============================
if __name__ == '__main__':
    # Example 1: your Portuguese sentence with a MEDICAL condition
    s = 'Bernardo, seu vesgo!, por que vc adora dormir de manhã e jogar videogame na Rua Santa Cecília?'
    pii = {
        'name': 'Bernardo',
        'medical_data': 'Bernardo é vesgo',  # narrative fuzzy
        'street': 'Rua Santa Cecília',
    }

    # Example 2: your JSON snippet (items_used-only)
    s2 = (
        'Paciente Aparecida Santos, nascida em 06/07/1996, possui CPF 68.479.146-5 e reside no CEP 99560-000. '
        'Dr. Richard registrou sinais vitais normais durante a consulta.'
    )
    pii2 = {
        'rg': '68.479.146-5',
        'surnames': 'Aparecida Santos',
        'cep': '99560-000',
        'name': 'Richard',
        'BIRTHDATE': '06/07/1996',
    }
