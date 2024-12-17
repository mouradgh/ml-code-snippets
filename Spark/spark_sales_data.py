# Handling retail data with Spark

# RetailWorld, a prominent retail chain with numerous stores across Metropolia,
# faces the challenge of processing and analyzing substantial volumes of daily sales data.
# With real-time data streaming from multiple sources, RetailWorld needs to clean,
# transform, and aggregate this data to derive actionable insights such as :
# total Sales and Revenue per Product, Total Sales and Revenue per Store,
# Sales and Revenue per Promotion Type and Stock Analysis per Product.

# FindSpark simplifies the process of using Apache Spark with Python
import findspark

findspark.init()

# Import SparkSession & SparkContext
from pyspark import SparkContext

# Initialize Spark context
sc = SparkContext(appName="RetailStoreSalesAnalysis")

# Load the CSV file
raw_data = sc.textFile("Data/Retailsales.csv")


# Parse and Clean Data
def parse_line(line):
    # Split the line by comma to get fields
    fields = line.split(",")
    # Return a dictionary with parsed fields
    return {
        "product_id": fields[0],
        "store_id": fields[1],
        "date": fields[2],
        "sales": float(fields[3]),
        "revenue": float(fields[4]),
        "stock": float(fields[5]),
        "price": float(fields[6]),
        "promo_type_1": fields[7],
        "promo_type_2": fields[9],
    }


# Remove the header line
header = raw_data.first()
raw_data_no_header = raw_data.filter(lambda line: line != header)

# Parse the lines into a structured format
parsed_data = raw_data_no_header.map(parse_line)

# Remove records with missing data
parsed_data = parsed_data.filter(lambda x: x is not None)

# Filter out records with invalid data
cleaned_data = parsed_data.filter(lambda x: x["sales"] > 0 and x["price"] > 0)

# Check the number of partitions
print(f"Number of partitions in cleaned_data: {cleaned_data.getNumPartitions()}")


# Function to count the number of records in each partition
def count_in_partition(index, iterator):
    count = sum(1 for _ in iterator)
    yield (index, count)


# Get the count of records in each partition
partitions_info = cleaned_data.mapPartitionsWithIndex(count_in_partition).collect()
print("Number of records in each partition:")
for partition, count in partitions_info:
    print(f"Partition {partition}: {count} records")

# Aggregation 1: Total Sales and Revenue per Product
# Map each record in cleaned_data to a key-value pair,
# where the key is the product ID and the value is a tuple containing sales and revenue.
# Then, reduceByKey to aggregate the sales and revenue values for each product ID
sales_revenue_per_product = cleaned_data.map(
    lambda x: (x["product_id"], (x["sales"], x["revenue"]))
).reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))

# Aggregation 2: Total Sales and Revenue per Store
# Similar to the first aggregation, map each record to a key-value pair with the store ID as the key
# and a tuple containing sales and revenue as the value.
# Then reduceByKey to aggregate the sales and revenue values for each store ID.
sales_revenue_per_store = cleaned_data.map(
    lambda x: (x["store_id"], (x["sales"], x["revenue"]))
).reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))

# Aggregation 3: Average Price per Product
# First map each record to a key-value pair with the product ID as the key
# and a tuple containing the price and a count of 1.
# Then reduceByKey to aggregate the total price and count of prices for each product.
# Finally, calculate the average price by dividing the total price by the count.
total_price_count_per_product = cleaned_data.map(
    lambda x: (x["product_id"], (x["price"], 1))
).reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
average_price_per_product = total_price_count_per_product.mapValues(
    lambda x: x[0] / x[1]
)

# Aggregation 4: Total Sales and Revenue per Promotion Type
# Each record is mapped to a key-value pair with the promotion type as the key
# and a tuple containing sales and revenue as the value.
# Then, reduceByKey is used to aggregate the sales and revenue values for each promotion type.
sales_revenue_per_promo_1 = cleaned_data.map(
    lambda x: (x["promo_type_1"], (x["sales"], x["revenue"]))
).reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
sales_revenue_per_promo_2 = cleaned_data.map(
    lambda x: (x["promo_type_2"], (x["sales"], x["revenue"]))
).reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))

# Aggregation 5: Stock Analysis per Product
# Each record is mapped to a key-value pair with the product ID as the key and the stock as the value.
# Then, reduceByKey is used to aggregate the stock values for each product
stock_per_product = cleaned_data.map(
    lambda x: (x["product_id"], x["stock"])
).reduceByKey(lambda a, b: a + b)

# Save the results to HDFS
sales_revenue_per_product.saveAsTextFile("RDD/sales_revenue_per_product")
sales_revenue_per_store.saveAsTextFile("RDD/sales_revenue_per_store")
average_price_per_product.saveAsTextFile("RDD/average_price_per_product")
sales_revenue_per_promo_1.saveAsTextFile("RDD/sales_revenue_per_promo_1")
sales_revenue_per_promo_2.saveAsTextFile("RDD/sales_revenue_per_promo_2")
stock_per_product.saveAsTextFile("RDD/stock_per_product")


# Print results
print("Total Sales and Revenue per Product:")
print("=" * 35)
for product in sales_revenue_per_product.collect():
    # Create the format string with appropriate padding
    format_string = "{:<5} | {:<9} | {:<9}"

    # Print the values using the format string
    print(
        format_string.format(
            str(product[0]), str(round(product[1][0], 2)), str(round(product[1][1], 2))
        )
    )

print("\n\nTotal Sales and Revenue per Store:")
print("=" * 35)
for store in sales_revenue_per_store.collect():
    format_string = "{:<5} | {:<9} | {:<9}"
    print(
        format_string.format(
            str(store[0]), str(round(store[1][0], 2)), str(round(store[1][1], 2))
        )
    )

print("\n\nAverage Price per Product:")
print("=" * 30)

for product in average_price_per_product.collect():
    format_string = "{:<5} | {:<9}"
    print(format_string.format(str(product[0]), str(round(product[1], 2))))

print("\n\nSales and Revenue per Promotion Type 1:")
print("=" * 40)
for promo in sales_revenue_per_promo_1.collect():
    format_string = "{:<5} | {:<9} | {:<9}"
    print(
        format_string.format(
            str(promo[0]), str(round(promo[1][0], 2)), str(round(promo[1][1], 2))
        )
    )

print("\n\nSales and Revenue per Promotion Type 2:")
print("=" * 40)
for promo in sales_revenue_per_promo_2.collect():
    format_string = "{:<5} | {:<9} | {:<9}"

    print(
        format_string.format(
            str(promo[0]), str(round(promo[1][0], 2)), str(round(promo[1][1], 2))
        )
    )

print("\n\nStock per Product:")
print("=" * 20)
for product in stock_per_product.collect():
    format_string = "{:<5} | {:<9}"
    print(format_string.format(str(product[0]), str(round(product[1], 2))))

# Stop the Spark context
sc.stop()
