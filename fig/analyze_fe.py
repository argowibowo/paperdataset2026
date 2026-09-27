import csv
import statistics
import math

file_path = 'dataset_hypersd200_labeled_featureenvy_withinstance.csv'
with open(file_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

# Representative metrics for Feature Envy
fe_metrics = ['LOC_method', 'CYCLO_method', 'MAXNESTING_method', 'NOP_method', 
              'NOAV_method', 'FANOUT_method', 'TCC_type', 'AMWNAMM_type',
              'ATFD_method', 'FDP_method', 'CM_method', 'CFNAMM_method',
              'LAA_method', 'WOC_type', 'CINT_method', 'CDISP_method']

print("=== DESCRIPTIVE STATISTICS FOR FE DATASET ===\n")
print(f"{'Metric':<20} {'Count':>8} {'Mean':>10} {'Std':>10} {'Min':>10} {'Median':>10} {'Max':>10}")
print("-" * 80)

for metric in fe_metrics:
    if metric in header:
        idx = header.index(metric)
        values = []
        for row in rows:
            try:
                val = float(row[idx])
                values.append(val)
            except:
                pass
        if values:
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0
            min_val = min(values)
            max_val = max(values)
            med_val = statistics.median(values)
            print(f"{metric:<20} {len(values):>8} {mean_val:>10.3f} {std_val:>10.3f} {min_val:>10.1f} {med_val:>10.1f} {max_val:>10.1f}")

# Correlation between some key pairs
print("\n=== KEY METRIC CORRELATIONS ===\n")
pairs = [
    ('CYCLO_method', 'CC_method'),
    ('LOC_method', 'CYCLO_method'),
    ('ATFD_method', 'FDP_method'),
    ('ATFD_method', 'FANOUT_method'),
    ('CINT_method', 'CDISP_method'),
    ('TCC_type', 'LAA_method'),
    ('WMC_type', 'WMCNAMM_type'),
]

def pearson_corr(x, y):
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
    if std_x == 0 or std_y == 0:
        return 0
    return cov / (std_x * std_y)

for m1, m2 in pairs:
    if m1 in header and m2 in header:
        idx1 = header.index(m1)
        idx2 = header.index(m2)
        vals1, vals2 = [], []
        for row in rows:
            try:
                v1 = float(row[idx1])
                v2 = float(row[idx2])
                vals1.append(v1)
                vals2.append(v2)
            except:
                pass
        if vals1:
            corr = pearson_corr(vals1, vals2)
            print(f"  {m1} -- {m2}: {corr:.3f}")

# Range validation
print("\n=== RANGE VALIDATION ===\n")
range_metrics = {
    'TCC_type': (0, 1),
    'WOC_type': (0, 1),
    'LAA_method': (0, 1),
    'CFNAMM_method': (0, 1),
    'LOC_method': (1, None),
    'CYCLO_method': (1, None),
    'MAXNESTING_method': (0, None),
}

for metric, (low, high) in range_metrics.items():
    if metric in header:
        idx = header.index(metric)
        violations = 0
        total = 0
        for row in rows:
            try:
                val = float(row[idx])
                total += 1
                if low is not None and val < low:
                    violations += 1
                if high is not None and val > high:
                    violations += 1
            except:
                pass
        status = "PASS" if violations == 0 else f"FAIL ({violations} violations)"
        expected = f"{low}--{high}" if high else f">={low}"
        print(f"  {metric:<20} Expected: {expected:<10} -> {status}")

print("\n=== CLASS DISTRIBUTION ===")
label_idx = header.index('is_feature_envy')
dist = {}
for row in rows:
    val = row[label_idx]
    dist[val] = dist.get(val, 0) + 1
total = sum(dist.values())
for k, v in sorted(dist.items()):
    print(f"  {k}: {v} ({v/total*100:.2f}%)")
print(f"  Total: {total}")
