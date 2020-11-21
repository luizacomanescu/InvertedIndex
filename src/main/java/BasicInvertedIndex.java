/**
 * Basic Inverted Index
 * <p>
 * This Map Reduce program should build an Inverted Index from a set of files.
 * Each token (the key) in a given file should reference the file it was found
 * in.
 * <p>
 * The output of the program should look like this:
 * sometoken [file001, file002, ... ]
 *
 * @author Kristian Epps
 */

import org.apache.commons.cli.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

import java.io.IOException;
import java.util.*;

public class BasicInvertedIndex extends Configured implements Tool {
    private static final Logger LOG = Logger
            .getLogger(BasicInvertedIndex.class);

    private static final int DUCUMENTS_COUNT = 5;

    public static class Map extends
            Mapper<Object, Text, Text, Text> {

        // INPUTFILE holds the name of the current file
        private final static Text INPUTFILE = new Text();

        // TOKEN should be set to the current token rather than creating a
        // new Text object for each one
        @SuppressWarnings("unused")
        private final static Text TOKEN = new Text();

        private final static IntWritable ONE = new IntWritable(1);
        private final static Text WORD = new Text();

        // The StopAnalyser class helps remove stop words
        @SuppressWarnings("unused")
        private StopAnalyser stopAnalyser = new StopAnalyser();

        private int fileLineCounter = 0;
        private int tokenPositionCounter = 0;
        private HashSet<Text> fileSet = new HashSet<>();
        private HashMap<String, ArrayListWritable<PairOfWritables<IntWritable, IntWritable>>> tokenMappings = new HashMap<>();

        // The stem method wraps the functionality of the Stemmer
        // class, which trims extra characters from English words
        // Please refer to the Stemmer class for more comments
        @SuppressWarnings("unused")
        private String stem(String word) {
            Stemmer s = new Stemmer();

            // A char[] word is added to the stemmer with its length,
            // then stemmed
            s.add(word.toCharArray(), word.length());
            s.stem();

            // return the stemmed char[] word as a string
            return s.toString();
        }

        // This method gets the name of the file the current Mapper is working
        // on
        @Override
        public void setup(Context context) {
            String inputFilePath = ((FileSplit) context.getInputSplit()).getPath().toString();
            String[] pathComponents = inputFilePath.split("/");
            INPUTFILE.set(pathComponents[pathComponents.length - 1]);
        }

        // TODO
        // This Mapper should read in a line, convert it to a set of tokens
        // and output each token with the name of the file it was found in
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {
            fileLineCounter++;

            String line = value.toString();
            StringTokenizer itr = new StringTokenizer(line);

            while (itr.hasMoreTokens()) {
                tokenPositionCounter++;

                String token = itr.nextToken();
                token = token.replaceAll("[^a-zA-Z]", "").toLowerCase();

                if (!StopAnalyser.isStopWord(token)) {
                    token = stem(token);

                    ArrayListWritable<PairOfWritables<IntWritable, IntWritable>> tokenOccurrenceList;

                    if (tokenMappings.containsKey(token)) {
                        tokenOccurrenceList = tokenMappings.get(token);
                        PairOfWritables<IntWritable, IntWritable> pair = new PairOfWritables<>();
                        pair.set(new IntWritable(fileLineCounter),
                                new IntWritable(tokenOccurrenceList.get(0).getRightElement().get() + 1));
                        tokenOccurrenceList.set(0, pair);
                    } else {
                        tokenOccurrenceList = new ArrayListWritable<>();
                        PairOfWritables<IntWritable, IntWritable> pair = new PairOfWritables<>();
                        pair.set(new IntWritable(fileLineCounter), ONE);
                        tokenOccurrenceList.add(pair);
                    }

                    tokenOccurrenceList.add(new PairOfWritables<>(new IntWritable(fileLineCounter), new IntWritable(tokenPositionCounter)));
                    tokenMappings.put(token, tokenOccurrenceList);
                }
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            for (java.util.Map.Entry<String, ArrayListWritable<PairOfWritables<IntWritable,
                    IntWritable>>> token : tokenMappings.entrySet()) {
                WORD.set(token.getKey());
                context.write(WORD, new PairOfWritables<>(INPUTFILE, token.getValue()));
            }

            fileSet.add(INPUTFILE);
        }
    }

    public static class Reduce extends Reducer<Text, PairOfWritables<Text,
            ArrayListWritable<PairOfWritables<IntWritable, IntWritable>>>,
            Text, ArrayListWritable<PairOfWritables<PairOfWritables<IntWritable,ArrayListWritable<FloatWritable>>,
            ArrayListWritable<PairOfWritables<PairOfWritables<Text, IntWritable>,
            ArrayListWritable<PairOfWritables<IntWritable, IntWritable>>>>>>> {

        // TODO
        // This Reduce Job should take in a key and an iterable of file names
        // It should convert this iterable to a writable array list and output
        // it along with the key
        public void reduce(
                Text key,
                Iterable<PairOfWritables<Text, ArrayListWritable<PairOfWritables
                        <IntWritable, IntWritable>>>> values,
                Context context) throws IOException, InterruptedException {

            Iterator<PairOfWritables<Text, ArrayListWritable<PairOfWritables
                    <IntWritable, IntWritable>>>> itr = values.iterator();
            ArrayListWritable<PairOfWritables<PairOfWritables<Text, IntWritable>, ArrayListWritable<PairOfWritables
                    <IntWritable, IntWritable>>>> tokenPostings = new ArrayListWritable<>();

            ArrayList<IntWritable> tokenFrequencyList = new ArrayList<>();

            while (itr.hasNext()) {

                PairOfWritables<Text, ArrayListWritable<PairOfWritables
                        <IntWritable, IntWritable>>> tokenOccurrenceList = itr.next();
                PairOfWritables<Text, IntWritable> tokenFrequency =
                        new PairOfWritables<>(tokenOccurrenceList.getLeftElement(),
                                tokenOccurrenceList.getRightElement().get(0).getRightElement());

                tokenFrequencyList.add(tokenOccurrenceList.getRightElement().get(0).getRightElement());
                tokenOccurrenceList.getRightElement().remove(0);

                tokenPostings.add(new PairOfWritables<>(tokenFrequency, tokenOccurrenceList.getRightElement()));
            }

            ArrayListWritable<FloatWritable> tfidfList = new ArrayListWritable<>();
            for(int i = 0; i < tokenFrequencyList.size(); i++) {
                float tfidf = (float) ((1 + Math.log10(tokenFrequencyList.get(i).get())) * Math.log10(DUCUMENTS_COUNT/ (float) tokenPostings.size()));
                tfidfList.add(new FloatWritable(tfidf));
            }

            ArrayListWritable<PairOfWritables<PairOfWritables<IntWritable,ArrayListWritable<FloatWritable>>, ArrayListWritable<PairOfWritables<PairOfWritables<Text, IntWritable>,
                    ArrayListWritable<PairOfWritables<IntWritable, IntWritable>>>>>> output = new ArrayListWritable<>();
            PairOfWritables<PairOfWritables<IntWritable,ArrayListWritable<FloatWritable>>, ArrayListWritable<PairOfWritables<PairOfWritables<Text, IntWritable>,
                    ArrayListWritable<PairOfWritables<IntWritable, IntWritable>>>>> documentFrequencyPair =
                    new PairOfWritables<>(new PairOfWritables<>(new IntWritable(tokenPostings.size()),tfidfList),tokenPostings);
            output.add(documentFrequencyPair);

            context.write(key, output);
        }
    }

    // Lets create an object! :)
    public BasicInvertedIndex() {
    }

    // Variables to hold cmd line args
    private static final String INPUT = "input";
    private static final String OUTPUT = "output";
    private static final String NUM_REDUCERS = "numReducers";

    @SuppressWarnings({"static-access"})
    public int run(String[] args) throws Exception {

        // Handle command line args
        Options options = new Options();
        options.addOption(OptionBuilder.withArgName("path").hasArg()
                .withDescription("input path").create(INPUT));
        options.addOption(OptionBuilder.withArgName("path").hasArg()
                .withDescription("output path").create(OUTPUT));
        options.addOption(OptionBuilder.withArgName("num").hasArg()
                .withDescription("number of reducers").create(NUM_REDUCERS));

        CommandLine cmdline = null;
        CommandLineParser parser = new uk.ac.man.cs.comp38211.util.XParser(true);

        try {
            cmdline = parser.parse(options, args);
        } catch (ParseException exp) {
            System.err.println("Error parsing command line: "
                    + exp.getMessage());
            System.err.println(cmdline);
            return -1;
        }

        // If we are missing the input or output flag, let the user know
        if (!cmdline.hasOption(INPUT) || !cmdline.hasOption(OUTPUT)) {
            System.out.println("args: " + Arrays.toString(args));
            HelpFormatter formatter = new HelpFormatter();
            formatter.setWidth(120);
            formatter.printHelp(this.getClass().getName(), options);
            ToolRunner.printGenericCommandUsage(System.out);
            return -1;
        }

        // Create a new Map Reduce Job
        Configuration conf = new Configuration();
        Job job = new Job(conf);
        String inputPath = cmdline.getOptionValue(INPUT);
        String outputPath = cmdline.getOptionValue(OUTPUT);
        int reduceTasks = cmdline.hasOption(NUM_REDUCERS) ? Integer
                .parseInt(cmdline.getOptionValue(NUM_REDUCERS)) : 1;

        // Set the name of the Job and the class it is in
        job.setJobName("Basic Inverted Index");
        job.setJarByClass(BasicInvertedIndex.class);
        job.setNumReduceTasks(reduceTasks);

        // Set the Mapper and Reducer class (no need for combiner here)
        job.setMapperClass(Map.class);
        job.setReducerClass(Reduce.class);

        // Set the Output Classes
        job.setMapOutputKeyClass(Text.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(ArrayListWritable.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(PairOfWritables.class);

        // Set the input and output file paths
        FileInputFormat.setInputPaths(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));

        // Time the job whilst it is running
        long startTime = System.currentTimeMillis();
        job.waitForCompletion(true);
        LOG.info("Job Finished in " + (System.currentTimeMillis() - startTime)
                / 1000.0 + " seconds");

        // Returning 0 lets everyone know the job was successful
        return 0;
    }

    public static void main(String[] args) throws Exception {
        ToolRunner.run(new BasicInvertedIndex(), args);
    }
}